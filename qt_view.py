import argparse
import math
import os
import sys
import time
from threading import Thread
from math import sqrt

import google.protobuf.timestamp_pb2
import numpy as np
import numpy.linalg
import vtk
from vtk.util import numpy_support

import graph_nav_util
import dijkstra

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.util import seconds_to_duration
from bosdyn.client import math_helpers, ResponseError, RpcError, create_standard_sdk
from bosdyn.client.math_helpers import Quat, Vec3
from bosdyn import geometry
from bosdyn.api import geometry_pb2 as geo, image_pb2, trajectory_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient, 
                                         blocking_stand, blocking_sit, 
                                         CommandFailedError, CommandTimedOutError)
from bosdyn.client.async_tasks import AsyncPeriodicQuery, AsyncTasks
from bosdyn.client.frame_helpers import (GROUND_PLANE_FRAME_NAME, VISION_FRAME_NAME, 
                                         GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME,
                                         get_a_tform_b,
                                         get_vision_tform_body,
                                         get_odom_tform_body)

from bosdyn.client.robot_command import (CommandFailedError, CommandTimedOutError,
                                         RobotCommandBuilder, RobotCommandClient, 
                                         blocking_stand, blocking_sit, blocking_command)
from bosdyn.api import geometry_pb2
from bosdyn.api.graph_nav import map_pb2
from bosdyn.client.frame_helpers import *
from bosdyn.client.math_helpers import *
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_state import RobotStateClient

from bosdyn.client.recording import GraphNavRecordingServiceClient

import google.protobuf.timestamp_pb2
from google.protobuf import wrappers_pb2 as wrappers

import bosdyn.client.channel
from bosdyn.api.graph_nav import map_pb2, map_processing_pb2, recording_pb2, graph_nav_pb2, nav_pb2
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.map_processing import MapProcessingServiceClient
from bosdyn.client.power import PowerClient


# PyQt5関連のインポート
from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame, QVBoxLayout, QToolTip, QMessageBox  # 変更: QMessageBox を追加
from PyQt5.QtGui import QCursor  # 変更: QCursorを追加
from PyQt5.QtCore import Qt, QTimer  # 変更: QTimerを追加

# QVTKRenderWindowInteractor のインポート
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# view_map.py 内の関数を利用するためにパス設定（既存の view_map.py は以下のディレクトリにある前提）
sys.path.append(os.path.join(os.path.dirname(__file__), 'python', 'examples', 'graph_nav_view_map'))
# from view_map import load_map, create_graph_objects, create_anchored_graph_objects, mat_to_vtk

# 追加: 各waypointのワールド座標を格納するグローバル辞書
WAYPOINT_POSITIONS = {}
WAYPOINT_NAMES = {}  # 新規追加: waypoint の name を保存する辞書

# グローバル辞書として追加
WAYPOINT_TEXT_ACTORS = {}

LOGGER = bosdyn.client.util.get_logger()

SPEED_LIMIT = 1.2
# CLIENT = {
#     "state": None,
#     "command": None,
#     "lease": None,
# }

def numpy_to_poly_data(pts):
    """
    Converts numpy array data into vtk poly data.
    :param pts: the numpy array to convert (3 x N).
    :return: a vtkPolyData.
    """
    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk.vtkPoints())
    # Makes a deep copy
    pd.GetPoints().SetData(numpy_support.numpy_to_vtk(pts.copy()))

    f = vtk.vtkVertexGlyphFilter()
    f.SetInputData(pd)
    f.Update()
    pd = f.GetOutput()

    return pd


def mat_to_vtk(mat):
    """
    Converts a 4x4 homogenous transform into a vtk transform object.
    :param mat: A 4x4 homogenous transform (numpy array).
    :return: A VTK transform object representing the transform.
    """
    t = vtk.vtkTransform()
    t.SetMatrix(mat.flatten())
    return t


def vtk_to_mat(transform):
    """
    Converts a VTK transform object to 4x4 homogenous numpy matrix.
    :param transform: an object of type vtkTransform
    : return: a numpy array with a 4x4 matrix representation of the transform.
    """
    tf_matrix = transform.GetMatrix()
    out = np.array(np.eye(4))
    for r in range(4):
        for c in range(4):
            out[r, c] = tf_matrix.GetElement(r, c)
    return out


def api_to_vtk_se3_pose(se3_pose):
    """
    Convert a bosdyn SDK SE3Pose into a VTK pose.
    :param se3_pose: the bosdyn SDK SE3 Pose.
    :return: A VTK pose representing the bosdyn SDK SE3 Pose.
    """
    return mat_to_vtk(se3_pose.to_matrix())


def create_fiducial_object(world_object, waypoint, renderer):
    """
    Creates a VTK object representing a fiducial.
    :param world_object: A WorldObject representing a fiducial.
    :param waypoint: The waypoint the AprilTag is associated with.
    :param renderer: The VTK renderer
    :return: a tuple of (vtkActor, 4x4 homogenous transform) representing the vtk actor for the fiducial, and its
    transform w.r.t the waypoint.
    """
    fiducial_object = world_object.apriltag_properties
    odom_tform_fiducial_filtered = get_a_tform_b(
        world_object.transforms_snapshot, ODOM_FRAME_NAME,
        world_object.apriltag_properties.frame_name_fiducial_filtered)
    waypoint_tform_odom = SE3Pose.from_proto(waypoint.waypoint_tform_ko)
    waypoint_tform_fiducial_filtered = api_to_vtk_se3_pose(
        waypoint_tform_odom * odom_tform_fiducial_filtered)
    plane_source = vtk.vtkPlaneSource()
    plane_source.SetCenter(0.0, 0.0, 0.0)
    plane_source.SetNormal(0.0, 0.0, 1.0)
    plane_source.Update()
    plane = plane_source.GetOutput()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(plane)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.5, 0.7, 0.9)
    actor.SetScale(fiducial_object.dimensions.x, fiducial_object.dimensions.y, 1.0)
    renderer.AddActor(actor)
    return actor, waypoint_tform_fiducial_filtered


def create_point_cloud_object(waypoints, snapshots, waypoint_id):
    """
    Create a VTK object representing the point cloud in a snapshot. Note that in graph_nav, "point cloud" refers to the
    feature cloud of a waypoint -- that is, a collection of visual features observed by all five cameras at a particular
    point in time. The visual features are associated with points that are rigidly attached to a waypoint.
    :param waypoints: dict of waypoint ID to waypoint.
    :param snapshots: dict of waypoint snapshot ID to waypoint snapshot.
    :param waypoint_id: the waypoint ID of the waypoint whose point cloud we want to render.
    :return: a vtkActor containing the point cloud data.
    """
    wp = waypoints[waypoint_id]
    snapshot = snapshots[wp.snapshot_id]
    cloud = snapshot.point_cloud
    odom_tform_cloud = get_a_tform_b(cloud.source.transforms_snapshot, ODOM_FRAME_NAME,
                                     cloud.source.frame_name_sensor)
    waypoint_tform_odom = SE3Pose.from_proto(wp.waypoint_tform_ko)
    waypoint_tform_cloud = api_to_vtk_se3_pose(waypoint_tform_odom * odom_tform_cloud)

    point_cloud_data = np.frombuffer(cloud.data, dtype=np.float32).reshape(int(cloud.num_points), 3)
    poly_data = numpy_to_poly_data(point_cloud_data)
    arr = vtk.vtkFloatArray()
    for i in range(cloud.num_points):
        arr.InsertNextValue(point_cloud_data[i, 2])
    arr.SetName('z_coord')
    poly_data.GetPointData().AddArray(arr)
    poly_data.GetPointData().SetActiveScalars('z_coord')
    actor = vtk.vtkActor()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    mapper.ScalarVisibilityOn()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(2)
    actor.SetUserTransform(waypoint_tform_cloud)
    return actor


def create_waypoint_object(renderer, waypoints, snapshots, waypoint_id):
    """
    Creates a VTK object representing a waypoint and its point cloud.
    :param renderer: The VTK renderer.
    :param waypoints: dict of waypoint ID to waypoint.
    :param snapshots: dict of snapshot ID to snapshot.
    :param waypoint_id: the waypoint id of the waypoint object we wish to create.
    :return: A vtkAssembly representing the waypoint (an axis) and its point cloud.
    """
    assembly = vtk.vtkAssembly()
    actor = vtk.vtkAxesActor()
    actor.SetXAxisLabelText('')
    actor.SetYAxisLabelText('')
    actor.SetZAxisLabelText('')
    actor.SetTotalLength(0.2, 0.2, 0.2)
    assembly.AddPart(actor)
    try:
        point_cloud_actor = create_point_cloud_object(waypoints, snapshots, waypoint_id)
        assembly.AddPart(point_cloud_actor)
    except Exception as e:
        print("Sorry, unable to create point cloud...", e)
    renderer.AddActor(assembly)
    return assembly


def make_line(pt_A, pt_B, renderer):
    """
    Creates a VTK object which is a white line between two points.
    :param pt_A: starting point of the line.
    :param pt_B: ending point of the line.
    :param renderer: the VTK renderer.
    :return: A VTK object that is a while line between pt_A and pt_B.
    """
    line_source = vtk.vtkLineSource()
    line_source.SetPoint1(pt_A[0], pt_A[1], pt_A[2])
    line_source.SetPoint2(pt_B[0], pt_B[1], pt_B[2])
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(line_source.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(2)
    actor.GetProperty().SetColor(0.7, 0.7, 0.7)
    renderer.AddActor(actor)
    return actor


def make_text(name, pt, renderer):
    """
    Creates white text on a black background at a particular point.
    :param name: The text to display.
    :param pt: The point in the world where the text will be displayed.
    :param renderer: The VTK renderer
    :return: the vtkActor representing the text.
    """
    actor = vtk.vtkTextActor()
    actor.SetInput(name)
    prop = actor.GetTextProperty()
    prop.SetBackgroundColor(0.0, 0.0, 0.0)
    prop.SetBackgroundOpacity(0.5)
    prop.SetFontSize(16)
    coord = actor.GetPositionCoordinate()
    coord.SetCoordinateSystemToWorld()
    coord.SetValue((pt[0], pt[1], pt[2]))

    renderer.AddActor(actor)
    return actor

def make_waypoint_text(name, pt, renderer, curr_name):
    if name == curr_name:
        actor = vtk.vtkTextActor()
        actor.SetInput(name)
        prop = actor.GetTextProperty()
        prop.SetBackgroundColor(0.0, 0.0, 0.0)
        prop.SetBackgroundOpacity(0.5)
        prop.SetFontSize(30)
        coord = actor.GetPositionCoordinate()
        coord.SetCoordinateSystemToWorld()
        coord.SetValue((pt[0], pt[1], pt[2]))
    else:
        actor = vtk.vtkTextActor()
        actor.SetInput(name)
        prop = actor.GetTextProperty()
        prop.SetBackgroundColor(0.0, 0.0, 0.0)
        prop.SetBackgroundOpacity(0.5)
        prop.SetFontSize(16)
        coord = actor.GetPositionCoordinate()
        coord.SetCoordinateSystemToWorld()
        coord.SetValue((pt[0], pt[1], pt[2]))

    renderer.AddActor(actor)
    return actor


def create_edge_object(curr_wp_tform_to_wp, world_tform_curr_wp, renderer):
    # Concatenate the edge transform.
    world_tform_to_wp = np.dot(world_tform_curr_wp, curr_wp_tform_to_wp)
    # Make a line between the current waypoint and the neighbor.
    make_line(world_tform_curr_wp[:3, 3], world_tform_to_wp[:3, 3], renderer)
    return world_tform_to_wp


def load_map(path):
    """
    Load a map from the given file path.
    :param path: Path to the root directory of the map.
    :return: the graph, waypoints, waypoint snapshots and edge snapshots.
    """
    with open(os.path.join(path, 'graph'), 'rb') as graph_file:
        # Load the graph file and deserialize it. The graph file is a protobuf containing only the waypoints and the
        # edges between them.
        data = graph_file.read()
        current_graph = map_pb2.Graph()
        current_graph.ParseFromString(data)

        # Set up maps from waypoint ID to waypoints, edges, snapshots, etc.
        current_waypoints = {}
        current_waypoint_snapshots = {}
        current_edge_snapshots = {}
        current_anchors = {}
        current_anchored_world_objects = {}

        # Load the anchored world objects first so we can look in each waypoint snapshot as we load it.
        for anchored_world_object in current_graph.anchoring.objects:
            current_anchored_world_objects[anchored_world_object.id] = (anchored_world_object,)
        # For each waypoint, load any snapshot associated with it.
        for waypoint in current_graph.waypoints:
            current_waypoints[waypoint.id] = waypoint

            if len(waypoint.snapshot_id) == 0:
                continue
            # Load the snapshot. Note that snapshots contain all of the raw data in a waypoint and may be large.
            file_name = os.path.join(path, 'waypoint_snapshots', waypoint.snapshot_id)
            if not os.path.exists(file_name):
                continue
            with open(file_name, 'rb') as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                try:
                    waypoint_snapshot.ParseFromString(snapshot_file.read())
                    current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
                except Exception as e:
                    print(f"{e}: {file_name}")

                for fiducial in waypoint_snapshot.objects:
                    if not fiducial.HasField('apriltag_properties'):
                        continue

                    str_id = str(fiducial.apriltag_properties.tag_id)
                    if (str_id in current_anchored_world_objects and
                            len(current_anchored_world_objects[str_id]) == 1):

                        # Replace the placeholder tuple with a tuple of (wo, waypoint, fiducial).
                        anchored_wo = current_anchored_world_objects[str_id][0]
                        current_anchored_world_objects[str_id] = (anchored_wo, waypoint, fiducial)

        # Similarly, edges have snapshot data.
        for edge in current_graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            file_name = os.path.join(path, 'edge_snapshots', edge.snapshot_id)
            if not os.path.exists(file_name):
                continue
            with open(file_name, 'rb') as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                current_edge_snapshots[edge_snapshot.id] = edge_snapshot
        for anchor in current_graph.anchoring.anchors:
            current_anchors[anchor.id] = anchor
        print(
            f'Loaded graph with {len(current_graph.waypoints)} waypoints, {len(current_graph.edges)} edges, '
            f'{len(current_graph.anchoring.anchors)} anchors, and {len(current_graph.anchoring.objects)} anchored world objects'
        )
        return (current_graph, current_waypoints, current_waypoint_snapshots,
                current_edge_snapshots, current_anchors, current_anchored_world_objects)


def create_anchored_graph_objects(current_graph, current_waypoint_snapshots, current_waypoints,
                                  current_anchors, current_anchored_world_objects, renderer,
                                  hide_waypoint_text, hide_world_object_text):
    """
    Creates all the VTK objects associated with the graph, in seed frame, if they are anchored.
    :param current_graph: the graph to use.
    :param current_waypoint_snapshots: dict from snapshot id to snapshot.
    :param current_waypoints: dict from waypoint id to waypoint.
    :param renderer: The VTK renderer
    :param hide_waypoint_text: whether text representing each waypoint should be hidden
    :param hide_world_object_text: whether text representing each world object should be hidden
    :return: the average position in world space of all the waypoints.
    """
    global WAYPOINT_POSITIONS
    global WAYPOINT_NAMES
    WAYPOINT_POSITIONS = {}
    WAYPOINT_NAMES = {}
    avg_pos = np.array([0.0, 0.0, 0.0])
    waypoints_in_anchoring = 0
    # Create VTK objects associated with each waypoint.
    for waypoint in current_graph.waypoints:
        if waypoint.id in current_anchors:
            waypoint_object = create_waypoint_object(renderer, current_waypoints,
                                                     current_waypoint_snapshots, waypoint.id)
            seed_tform_waypoint = SE3Pose.from_proto(
                current_anchors[waypoint.id].seed_tform_waypoint).to_matrix()
            waypoint_object.SetUserTransform(mat_to_vtk(seed_tform_waypoint))
            # 更新: waypointの位置（平行移動部分）を保存
            WAYPOINT_POSITIONS[waypoint.id] = seed_tform_waypoint[:3, 3]
            WAYPOINT_NAMES[waypoint.id] = waypoint.annotations.name  # 新規追加: waypoint の name を保存
            if not hide_waypoint_text:
                make_text(waypoint.annotations.name, seed_tform_waypoint[:3, 3], renderer)
            avg_pos += seed_tform_waypoint[:3, 3]
            waypoints_in_anchoring += 1

    avg_pos /= waypoints_in_anchoring

    # Create VTK objects associated with each edge.
    for edge in current_graph.edges:
        if edge.id.from_waypoint in current_anchors and edge.id.to_waypoint in current_anchors:
            seed_tform_from = SE3Pose.from_proto(
                current_anchors[edge.id.from_waypoint].seed_tform_waypoint).to_matrix()
            from_tform_to = SE3Pose.from_proto(edge.from_tform_to).to_matrix()
            create_edge_object(from_tform_to, seed_tform_from, renderer)

    # Create VTK objects associated with each anchored world object.
    for anchored_wo in current_anchored_world_objects.values():
        # anchored_wo is a tuple of (anchored_world_object, waypoint, fiducial).
        (fiducial_object, _) = create_fiducial_object(anchored_wo[2], anchored_wo[1], renderer)
        seed_tform_fiducial = SE3Pose.from_proto(anchored_wo[0].seed_tform_object).to_matrix()
        fiducial_object.SetUserTransform(mat_to_vtk(seed_tform_fiducial))
        if not hide_world_object_text:
            make_text(anchored_wo[0].id, seed_tform_fiducial[:3, 3], renderer)

    return avg_pos


def create_graph_objects(current_graph, current_waypoint_snapshots, current_waypoints, renderer,
                         hide_waypoint_text, hide_world_object_text, curr_name):
    """
    Creates all the VTK objects associated with the graph.
    :param current_graph: the graph to use.
    :param current_waypoint_snapshots: dict from snapshot id to snapshot.
    :param current_waypoints: dict from waypoint id to waypoint.
    :param renderer: The VTK renderer
    :param hide_waypoint_text: whether text representing each waypoint should be hidden
    :param hide_world_object_text: whether text representing each world object should be hidden
    :return: the average position in world space of all the waypoints.
    """
    waypoint_objects = {}
    # Create VTK objects associated with each waypoint.
    for waypoint in current_graph.waypoints:
        waypoint_objects[waypoint.id] = create_waypoint_object(renderer, current_waypoints,
                                                               current_waypoint_snapshots,
                                                               waypoint.id)
    global WAYPOINT_POSITIONS
    global WAYPOINT_NAMES
    WAYPOINT_POSITIONS = {}
    WAYPOINT_NAMES = {}
    # Now, perform a breadth first search of the graph starting from an arbitrary waypoint. Graph nav graphs
    # have no global reference frame. The only thing we can say about waypoints is that they have relative
    # transformations to their neighbors via edges. So the goal is to get the whole graph into a global reference
    # frame centered on some waypoint as the origin.
    queue = []
    queue.append((current_graph.waypoints[0], np.eye(4)))
    visited = {}
    # Get the camera in the ballpark of the right position by centering it on the average position of a waypoint.
    avg_pos = np.array([0.0, 0.0, 0.0])

    # Breadth first search.
    while len(queue) > 0:
        # Visit a waypoint.
        curr_element = queue[0]
        queue.pop(0)
        curr_waypoint = curr_element[0]
        if curr_waypoint.id in visited:
            continue
        visited[curr_waypoint.id] = True

        # We now know the global pose of this waypoint, so set the pose.
        waypoint_objects[curr_waypoint.id].SetUserTransform(mat_to_vtk(curr_element[1]))
        world_tform_current_waypoint = curr_element[1]
        # 更新: waypointの位置を保存
        WAYPOINT_POSITIONS[curr_waypoint.id] = world_tform_current_waypoint[:3, 3]
        WAYPOINT_NAMES[curr_waypoint.id] = curr_waypoint.annotations.name  # 新規追加: waypoint の name を保存
        # Add text to the waypoint.
        if not hide_waypoint_text:
            # make_text(curr_waypoint.annotations.name, world_tform_current_waypoint[:3, 3], renderer)
            actor = make_waypoint_text(curr_waypoint.annotations.name, world_tform_current_waypoint[:3, 3], renderer, curr_name)
            WAYPOINT_TEXT_ACTORS[curr_waypoint.id] = actor

        # For each fiducial in the waypoint's snapshot, add an object at the world pose of that fiducial.
        if curr_waypoint.snapshot_id in current_waypoint_snapshots:
            snapshot = current_waypoint_snapshots[curr_waypoint.snapshot_id]
            for fiducial in snapshot.objects:
                if fiducial.HasField('apriltag_properties'):
                    (fiducial_object, curr_wp_tform_fiducial) = create_fiducial_object(
                        fiducial, curr_waypoint, renderer)
                    world_tform_fiducial = np.dot(world_tform_current_waypoint,
                                                  vtk_to_mat(curr_wp_tform_fiducial))
                    fiducial_object.SetUserTransform(mat_to_vtk(world_tform_fiducial))
                    if not hide_world_object_text:
                        make_text(str(fiducial.apriltag_properties.tag_id),
                                  world_tform_fiducial[:3, 3], renderer)

        # Now, for each edge, walk along the edge and concatenate the transform to the neighbor.
        for edge in current_graph.edges:
            # If the edge is directed away from us...
            if edge.id.from_waypoint == curr_waypoint.id and edge.id.to_waypoint not in visited:
                current_waypoint_tform_to_waypoint = SE3Pose.from_proto(
                    edge.from_tform_to).to_matrix()
                world_tform_to_wp = create_edge_object(current_waypoint_tform_to_waypoint,
                                                       world_tform_current_waypoint, renderer)
                # Add the neighbor to the queue.
                queue.append((current_waypoints[edge.id.to_waypoint], world_tform_to_wp))
                avg_pos += world_tform_to_wp[:3, 3]
            # If the edge is directed toward us...
            elif edge.id.to_waypoint == curr_waypoint.id and edge.id.from_waypoint not in visited:
                current_waypoint_tform_from_waypoint = (SE3Pose.from_proto(
                    edge.from_tform_to).inverse()).to_matrix()
                world_tform_from_wp = create_edge_object(current_waypoint_tform_from_waypoint,
                                                         world_tform_current_waypoint, renderer)
                # Add the neighbor to the queue.
                queue.append((current_waypoints[edge.id.from_waypoint], world_tform_from_wp))
                avg_pos += world_tform_from_wp[:3, 3]

    # Compute the average waypoint position to place the camera appropriately.
    avg_pos /= len(current_waypoints)
    return avg_pos

class AsyncRobotState(AsyncPeriodicQuery):
    """Grab robot state."""

    def __init__(self, robot_state_client):
        # period is set to be about the same rate as detections on the CORE AI
        super(AsyncRobotState, self).__init__('robot_state', robot_state_client, LOGGER,
                                              period_sec=0.02)

    def _start_query(self):
        return self._client.get_robot_state_async()
    
class GraphNavInterface(object):
    """GraphNav service command line interface."""

    def __init__(self, robot, upload_path, use_gps=False):
        self._robot = robot
        self.use_gps = use_gps

        # Force trigger timesync.
        self._robot.time_sync.wait_for_sync()

        # Create robot state and command clients.
        self._robot_command_client = self._robot.ensure_client(
            RobotCommandClient.default_service_name)
        self._robot_state_client = self._robot.ensure_client(RobotStateClient.default_service_name)

        # Create the client for the Graph Nav main service.
        self._graph_nav_client = self._robot.ensure_client(GraphNavClient.default_service_name)

        # Create a power client for the robot.
        self._power_client = self._robot.ensure_client(PowerClient.default_service_name)

        # Boolean indicating the robot's power state.
        power_state = self._robot_state_client.get_robot_state().power_state
        self._started_powered_on = (power_state.motor_power_state == power_state.STATE_ON)
        self._powered_on = self._started_powered_on

        # Number of attempts to wait before trying to re-power on.
        self._max_attempts_to_wait = 50

        # Store the most recent knowledge of the state of the robot based on rpc calls.
        self._current_graph = None
        self._current_edges = dict()  #maps to_waypoint to list(from_waypoint)
        self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
        self._current_edge_snapshots = dict()  # maps id to edge snapshot
        self._current_annotation_name_to_wp_id = dict()
        
        
        # waypointをidを格納する配列
        self._waypoint_all_id = []
        self._waypoint_all_name = []
        
        # 自動運転を制御するためのブール変数
        self._is_finished = True
        
        # マップはspotにアップロードされたか
        self._is_uploaded = False
        
        self._is_autodrive_cancel = False

        # Filepath for uploading a saved graph's and snapshots too.
        if upload_path[-1] == '/':
            self._upload_filepath = upload_path[:-1]
        else:
            self._upload_filepath = upload_path

        if self.use_gps:
            self._command_dictionary['g'] = self._navigate_to_gps_coords
            
    

    def _get_localization_state(self, *args):
        """Get the current localization and state of the robot."""
        state = self._graph_nav_client.get_localization_state(request_gps_state=self.use_gps)
        print(f'Got localization: \n{state.localization}')
        odom_tform_body = get_odom_tform_body(state.robot_kinematics.transforms_snapshot)
        print(f'Got robot state in kinematic odometry frame: \n{odom_tform_body}')
        if self.use_gps:
            print(f'GPS info:\n{state.gps}')

    def _set_initial_localization_fiducial(self, *args):
        """Trigger localization when near a fiducial."""
        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an empty instance for initial localization since we are asking it to localize
        # based on the nearest fiducial.
        localization = nav_pb2.Localization()
        self._graph_nav_client.set_localization(initial_guess_localization=localization,
                                                ko_tform_body=current_odom_tform_body)


    def _set_initial_localization_waypoint(self, *args):
        """Trigger localization to a waypoint."""
        # Take the first argument as the localization waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without initializing.
            print('No waypoint specified to initialize to.')
            return
        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            args[0][0], self._current_graph, self._current_annotation_name_to_wp_id)
        if not destination_waypoint:
            # Failed to find the unique waypoint id.
            return

        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an initial localization to the specified waypoint as the identity.
        localization = nav_pb2.Localization()
        localization.waypoint_id = destination_waypoint
        localization.waypoint_tform_body.rotation.w = 1.0
        self._graph_nav_client.set_localization(
            initial_guess_localization=localization,
            # It's hard to get the pose perfect, search +/-20 deg and +/-20cm (0.2m).
            max_distance=0.2,
            max_yaw=20.0 * math.pi / 180.0,
            fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
            ko_tform_body=current_odom_tform_body)
        
    def _get_localication_id(self):
        """現在、経路のどこにいるか（一番近くのwaypointを見つける）"""

        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print('Empty graph.')
            return None, None, None
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id
        
        for i in range(len(self._waypoint_all_id)):
            if self._waypoint_all_id[i] == localization_id:
                # print(f"現在のwaypointは{self._waypoint_all_name[i]}です")
                localization_name = self._waypoint_all_name[i]
                break
        
        print("現在のwaypoint_idは", localization_id)
        
        return localization_id, localization_name, i

    def _list_graph_waypoint_and_edge_ids(self, *args):
        """List the waypoint ids and edge ids of the graph currently on the robot."""

        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print('Empty graph.')
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id

        # Update and print waypoints and edges
        # self._current_annotation_name_to_wp_id, self._current_edges = graph_nav_util.update_waypoints_and_edges(
        #     graph, localization_id)
        
        # 作成順に並び替える（みつき作）
        self._current_annotation_name_to_wp_id, self._current_edges, self._waypoint_all_id, self._waypoint_all_name = graph_nav_util.update_waypoints_and_edges(
            graph, localization_id)
        
        print("waypointをidを出力します")
        for i in range(len(self._waypoint_all_id)):
            print(self._waypoint_all_id[i])
            
        print("waypointを名前を出力します")
        for i in range(len(self._waypoint_all_name)):
            print(self._waypoint_all_name[i])
            
        self._is_uploaded = True
        
    
    def _get_user_waypoint(self):
        
        # 作成したウェイポイントを探す
        # 作成していないと見つからない
        # 一番最後に作成したwaypointを探すようにしているはず
        if self._is_uploaded == False:
            print("マップがアップロードされていません")
            return None, None, None
        max_id = -1  # 最大の数値部分を保持する変数
        max_waypoint = None  # 最大のウェイポイント名を保持する変数
        max_index = -1  # 最大のウェイポイントのインデックスを保持する変数

        for i in range(len(self._waypoint_all_id)):
            if "created_user_waypoint" in self._waypoint_all_name[i]:
                # ウェイポイント名から数字部分を抽出
                number_part = self._waypoint_all_name[i][len("created_user_waypoint"):]

                try:
                    number_value = int(number_part)  # 数値に変換
                    if number_value > max_id:  # 最大値を更新
                        max_id = number_value
                        max_waypoint = self._waypoint_all_name[i]
                        max_index = i
                except ValueError:
                    # 数値変換に失敗した場合は無視
                    continue

        if max_waypoint is not None:
            print(f"{max_waypoint}: {self._waypoint_all_id[max_index]}")
            return max_waypoint, self._waypoint_all_id[max_index], max_index

        print("ユーザが作成したウェイポイントが見つかりません")
        return None, None, None
            

    def _upload_graph_and_snapshots(self, *args):
        """Upload the graph and snapshots to the robot."""
        print('Loading the graph from disk into local storage...')
        with open(self._upload_filepath + '/graph', 'rb') as graph_file:
            # Load the graph from disk.
            data = graph_file.read()
            self._current_graph = map_pb2.Graph()
            self._current_graph.ParseFromString(data)
            print(
                f'Loaded graph has {len(self._current_graph.waypoints)} waypoints and {self._current_graph.edges} edges'
            )
        for waypoint in self._current_graph.waypoints:
            # Load the waypoint snapshots from disk.
            with open(f'{self._upload_filepath}/waypoint_snapshots/{waypoint.snapshot_id}',
                      'rb') as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(snapshot_file.read())
                self._current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
        for edge in self._current_graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            # Load the edge snapshots from disk.
            with open(f'{self._upload_filepath}/edge_snapshots/{edge.snapshot_id}',
                      'rb') as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                self._current_edge_snapshots[edge_snapshot.id] = edge_snapshot
        # Upload the graph to the robot.
        print('Uploading the graph and snapshots to the robot...')
        true_if_empty = not len(self._current_graph.anchoring.anchors)
        response = self._graph_nav_client.upload_graph(graph=self._current_graph,
                                                       generate_new_anchoring=true_if_empty)
        # Upload the snapshots to the robot.
        for snapshot_id in response.unknown_waypoint_snapshot_ids:
            waypoint_snapshot = self._current_waypoint_snapshots[snapshot_id]
            self._graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
            print(f'Uploaded {waypoint_snapshot.id}')
        for snapshot_id in response.unknown_edge_snapshot_ids:
            edge_snapshot = self._current_edge_snapshots[snapshot_id]
            self._graph_nav_client.upload_edge_snapshot(edge_snapshot)
            print(f'Uploaded {edge_snapshot.id}')

        # The upload is complete! Check that the robot is localized to the graph,
        # and if it is not, prompt the user to localize the robot before attempting
        # any navigation commands.
        localization_state = self._graph_nav_client.get_localization_state()
        if not localization_state.localization.waypoint_id:
            # The robot is not localized to the newly uploaded graph.
            print('\n')
            print(
                'Upload complete! The robot is currently not localized to the map; please localize'
                ' the robot using commands (2) or (3) before attempting a navigation command.')



    def _navigate_to(self, *args):
        """Navigate to a specific waypoint."""
        # Take the first argument as the destination waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without requesting navigation.
            print('No waypoint provided as a destination for navigate to.')
            return

        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            args[0][0], self._current_graph, self._current_annotation_name_to_wp_id)
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the navigation command.
            return
        if not self.toggle_power(should_power_on=True):
            print('Failed to power on the robot, and cannot complete navigate to request.')
            return

        nav_to_cmd_id = None
        # Navigate to the destination waypoint.
        self._is_finished = False
        while not self._is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to(destination_waypoint, 1.0,
                                                                   command_id=nav_to_cmd_id)
            except ResponseError as e:
                print(f'Error while navigating {e}')
                break
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            self._is_finished = self._check_success(nav_to_cmd_id)

        # Power off the robot if appropriate.
        if self._powered_on and not self._started_powered_on:
            # Sit the robot down + power off after the navigation command is complete.
            self.toggle_power(should_power_on=False)
            
    def _navigate_first_waypoint(self, *args):
        # 移動先のwaypointを指定する
        destination_waypoint = args[0]
        print("destination_waypoint", destination_waypoint)
        
        if not destination_waypoint:
            print("waypointが記録されていない可能性があります")
            return
        
        nav_to_cmd_id = None
        # Navigate to the destination waypoint.
        self._is_finished = False
        while not self._is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to(destination_waypoint, 1.0,
                                                                   command_id=nav_to_cmd_id)
            except ResponseError as e:
                print(f'Error while navigating {e}')
                break
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            self._is_finished = self._check_success(nav_to_cmd_id)
        print("waypointに到着しました")
        # log_event("waypointに到着しました")
        # engine.say("arrived at the waypoint")
        # engine.runAndWait()
        # voice_output("arrived at the waypoint")
        print("-" * 80)

    def _navigate_route(self, *args):
        
        speed_x = 0.6 * SPEED_LIMIT
        speed_y = 1.0 * SPEED_LIMIT
        speed_angle = 0.5 * SPEED_LIMIT

        # 速度の設定
        speed_limit = geo.SE2VelocityLimit(
        max_vel=geo.SE2Velocity(
        linear=geo.Vec2(x=speed_x, y=speed_y),
        angular=speed_angle))
        print("確認1")
        
        params = self._graph_nav_client.generate_travel_params(
        max_distance = 0.5, max_yaw = 0.5, velocity_limit = speed_limit)

        print("確認2")
        
        waypoint_ids = self._waypoint_all_id
        for i in range(len(waypoint_ids)):
            print("before waypoint_ids[i]", waypoint_ids[i])
            waypoint_ids[i] = graph_nav_util.find_unique_waypoint_id(
                waypoint_ids[i], self._current_graph, self._current_annotation_name_to_wp_id)
            if not waypoint_ids[i]:
                # Failed to find the unique waypoint id.
                return
            print("after waypoint_ids[i]", waypoint_ids[i])
            
        edge_ids_list = []
        
        print("確認3")
        for i in range(len(waypoint_ids) - 1):
            start_wp = waypoint_ids[i]
            end_wp = waypoint_ids[i + 1]
            edge_id = self._match_edge(self._current_edges, start_wp, end_wp)
            if edge_id is not None:
                edge_ids_list.append(edge_id)
            else:
                all_edges_found = False
                print(f'Failed to find an edge between waypoints: {start_wp} and {end_wp}')
                print(
                    'List the graph\'s waypoints and edges to ensure pairs of waypoints has an edge.'
                )
                break
            
            
        route = self._graph_nav_client.build_route(waypoint_ids, edge_ids_list)
        self._is_finished = False
        previous_waypoint = None
        print("確認4")
        while not self._is_finished:
            try:
                nav_route_command_id = self._graph_nav_client.navigate_route(
                    route, cmd_duration=1.0, travel_params = params)

                # 終点に到達したかの確認
                self._is_finished = self._check_success(nav_route_command_id)
                
                if (self._is_finished):
                    print("終点に到達しました")
                    # engine.say("arrived at the end point")
                    # engine.runAndWait()
                    # voice_output("arrived at the end point")
                    print("-" * 80)
                    break
                    
                
            except ResponseError as e:
                break

            
            
    def _navigate_route_to_user_waypoint(self, current, current_waypoint_id, goal, goal_waypoint_id):
        # ユーザーが設定したwaypointに移動する
        
        # self.rcl._download_full_graph()
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print('Empty graph.')
            return
        self._current_graph = graph
        
        minrootgraph = dijkstra.MinRootGraph()
        
        # 入力データの作成
        graph_waypoints = []
        graph_edges = []
        
        print("確認1")
        for waipoint in range(len(self._current_graph.waypoints)):
            graph_dict = {
                "id": self._current_graph.waypoints[waipoint].id,
                "position": self._current_graph.waypoints[waipoint].waypoint_tform_ko.position,
                "name": self._current_graph.waypoints[waipoint].annotations.name
            }
            graph_waypoints.append(graph_dict)
            
        print("確認2")
            
        for edge in range(len(self._current_graph.edges)):
            value = self._current_graph.edges[edge].annotations.cost
            x = self._current_graph.edges[edge].from_tform_to.position.x
            y = self._current_graph.edges[edge].from_tform_to.position.y
            z = self._current_graph.edges[edge].from_tform_to.position.z
            
            position = sqrt(x**2 + y**2 + z**2)
            
            if hasattr(value, "value"):
                cost = float(value.value)
            else:
                cost = float(value)
            graph_dict = {
                "from_waypoint": self._current_graph.edges[edge].id.from_waypoint,
                "to_waypoint": self._current_graph.edges[edge].id.to_waypoint,
                "position": position,
            }
            print(graph_dict)
            graph_edges.append(graph_dict)

        print("確認3")
            
        for wayopint in range(len(graph_waypoints)):
            minrootgraph.add_waypoint(self._current_graph.waypoints[wayopint].id)
            
        for edge in range(len(self._current_graph.edges)):
            minrootgraph.add_edge(self._current_graph.edges[edge].id.from_waypoint, self._current_graph.edges[edge].id.to_waypoint, self._current_graph.edges[edge].annotations.cost)

        start_wp_id = current_waypoint_id
        goal_wp_id = goal_waypoint_id
        
        distances, path = minrootgraph.dijkstra(start_wp_id)
        
        print("確認4")
        def get_path(paths, start, end):
            path = []
            while end != start:
                path.append(end)
                end = paths[end]
            path.append(start)
            return path[::-1]
        
        shortest_path = get_path(path, start_wp_id, goal_wp_id)
        
        waypoint_route = [] 
        print("確認5")
        for i in range(len(shortest_path)):
            shortest_path[i] = graph_nav_util.find_unique_waypoint_id(
                shortest_path[i], self._current_graph, self._current_annotation_name_to_wp_id)
            if not shortest_path[i]:
                # Failed to find the unique waypoint id.
                return
            waypoint_route.append(shortest_path[i])
            
        edge_ids_list = []
        for i in range(len(waypoint_route) - 1):
            start_wp = waypoint_route[i]
            end_wp = waypoint_route[i + 1]
            edge_id = self._match_edge(self._current_edges, start_wp, end_wp)
            if edge_id is not None:
                edge_ids_list.append(edge_id)
            else:
                all_edges_found = False
                print(f'Failed to find an edge between waypoints: {start_wp} and {end_wp}')
                print(
                    'List the graph\'s waypoints and edges to ensure pairs of waypoints has an edge.'
                )
                break
        print("確認6")
        
        speed_x = 0.6 * SPEED_LIMIT
        speed_y = 1.0 * SPEED_LIMIT
        speed_angle = 0.5 * SPEED_LIMIT

        speed_limit = geo.SE2VelocityLimit(
        max_vel=geo.SE2Velocity(
        linear=geo.Vec2(x=speed_x, y=speed_y),
        angular=speed_angle))
        
        params = self._graph_nav_client.generate_travel_params(
        max_distance = 0.5, max_yaw = 0.5, velocity_limit = speed_limit)

            
        route = self._graph_nav_client.build_route(waypoint_route, edge_ids_list)
        self._is_finished = False
        self._is_autodrive_cancel = False
        
        while not self._is_finished:
            try:
                nav_route_command_id = self._graph_nav_client.navigate_route(
                    route, cmd_duration=1.4, travel_params = params)

                # 終点に到達したかの確認
                self._is_finished = self._check_success(nav_route_command_id)
                if (self._is_finished):
                    # engine.say("arrive at the goal")
                    # engine.runAndWait()
                    print("arrive at the goal")
                    # log_event("到着しました")
                    print("-" * 80)
                    # voice_output("arrive at the goal")
                    
                    try:
                        self._set_initial_localization_fiducial()
                    except Exception as e:
                        print(f'failed initialize {e}')
                    print("-" * 80)
                    
                    break
                
                # 自動運転がキャンセルされたときの処理
                if (self._is_autodrive_cancel):
                    print("自動運転をキャンセルしました")
                    # engine.say("cancel auto drive")
                    # engine.runAndWait()
                    print("-" * 80)
                    try:
                        self._set_initial_localization_fiducial()
                    except Exception as e:
                        print(f'failed initialize {e}')
                    print("-" * 80)
                    
                    self._is_finished = True
                    
                    break
                
            except ResponseError as e:
                self._is_finished = True
                print("-" * 80)
                break
        
            
            
    def _check_success(self, command_id=-1):
        """Use a navigation command id to get feedback from the robot and sit when command succeeds."""
        if command_id == -1:
            # No command, so we have no status to check.
            return False
        status = self._graph_nav_client.navigation_feedback(command_id)
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            # Successfully completed the navigation commands!
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            print('Robot got lost when navigating the route, the robot will now sit down.')
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            print('Robot got stuck when navigating the route, the robot will now sit down.')
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            print('Robot is impaired.')
            return True
        else:
            # Navigation command is not complete yet.
            return False
        
            
        
            
    def _match_edge(self, current_edges, waypoint1, waypoint2):
        """Find an edge in the graph that is between two waypoint ids."""
        # Return the correct edge id as soon as it's found.
        for edge_to_id in current_edges:
            for edge_from_id in current_edges[edge_to_id]:
                if (waypoint1 == edge_to_id) and (waypoint2 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint2, to_waypoint=waypoint1)
                elif (waypoint2 == edge_to_id) and (waypoint1 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint1, to_waypoint=waypoint2)
        return None

    def _clear_graph(self, *args):
        """Clear the state of the map on the robot, removing all waypoints and edges."""
        return self._graph_nav_client.clear_graph()

class MainWindow(QMainWindow):
    def __init__(self, map_path, anchoring, hide_wp_text, hide_wo_text, robot, client, parent=None):
        super(MainWindow, self).__init__(parent)
        
        self.gcl = GraphNavInterface(robot, "downloaded_graph", False)
        self.gcl._list_graph_waypoint_and_edge_ids()
        self.setWindowTitle("spot enviroment map")

        # QVTKRenderWindowInteractor ウィジェットの作成
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        frame = QFrame()
        layout = QVBoxLayout()
        layout.addWidget(self.vtk_widget)
        frame.setLayout(layout)
        self.setCentralWidget(frame)
        
        # 現在のwaypointを取得
        curr_id, curr_name, curr_i = self.gcl._get_localication_id()
        

        # VTK レンダラの作成とウィジェットに追加
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.05, 0.1, 0.15)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        # --- 変更箇所: RenderWindow サイズとインタラクタスタイルの設定 ---
        rw = self.vtk_widget.GetRenderWindow()
        rw.SetSize(1280, 720)
        interactor = rw.GetInteractor()
        style = vtk.vtkInteractorStyleTerrain()
        interactor.SetInteractorStyle(style)
        # --- ここまで変更箇所 ---

        # マップの読み込み
        (current_graph, current_waypoints, current_wp_snapshots, current_edge_snapshots,
         current_anchors, current_anchored_objects) = load_map(map_path)

        # グラフオブジェクトの作成（アンカリングの有無で処理を切り替え）
        if anchoring:
            if len(current_graph.anchoring.anchors) == 0:
                print("No anchors to draw.")
                sys.exit(-1)
            avg_pos = create_anchored_graph_objects(
                current_graph, current_wp_snapshots, current_waypoints, current_anchors,
                current_anchored_objects, self.renderer, hide_wp_text, hide_wo_text)
        else:
            avg_pos = create_graph_objects(
                current_graph, current_wp_snapshots, current_waypoints, self.renderer,
                hide_wp_text, hide_wo_text, curr_name)

        # カメラの設定
        camera_pos = avg_pos + np.array([-1, 0, 5])
        camera = self.renderer.GetActiveCamera()
        camera.SetViewUp(0, 0, 1)
        camera.SetPosition(camera_pos[0], camera_pos[1], camera_pos[2])
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.vtk_widget.Initialize()

        # 追加: クリックイベントのオブザーバー登録
        interactor.AddObserver("LeftButtonPressEvent", self.on_left_button_press)
        # --- 変更: 1秒ごとに現在の位置を更新するタイマーを追加 ---
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(1000)
        self.update_timer.timeout.connect(self.update_current_location)
        self.update_timer.start()

    def update_current_location(self):
        # 現在のローカライズ情報を取得
        curr_id, curr_name, _ = self.gcl._get_localication_id()
        # 各テキストアクターを更新：表示しているWaypoint名と現在のWaypoint名を比較
        for wp_id, actor in WAYPOINT_TEXT_ACTORS.items():
            # actorのテキストはSetInputで設定した内容（Waypoint名）
            current_text = actor.GetInput()
            if current_text == curr_name:
                actor.GetTextProperty().SetFontSize(30)
            else:
                actor.GetTextProperty().SetFontSize(16)
        # レンダラに更新を通知
        self.vtk_widget.GetRenderWindow().Render()

    def on_left_button_press(self, obj, event):
        # 空間のどこでもピックできるように vtkWorldPointPicker を使用する
        picker = vtk.vtkWorldPointPicker()
        click_pos = obj.GetEventPosition()  # (x, y)
        picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        pick_pos = np.array(picker.GetPickPosition())
        # 最近傍のwaypointを検索
        min_dist = float('inf')
        nearest_wp = None
        for wp_id, pos in WAYPOINT_POSITIONS.items():
            d = np.linalg.norm(np.array(pos) - pick_pos)
            if d < min_dist:
                min_dist = d
                nearest_wp = wp_id
        # 条件が真の時にYes/No のダイアログを表示
        if nearest_wp is not None and min_dist < 2.0:
            msg = f"Nearest waypoint: {nearest_wp}\nName: {WAYPOINT_NAMES.get(nearest_wp, 'N/A')}\nDistance: {min_dist:.2f}\nDo you want to select this waypoint?"
            msg_waypoint = f"Waypoint name: {WAYPOINT_NAMES.get(nearest_wp, 'N/A')}\n距離: {min_dist:.2f}\nこのWaypointを選択しますか？"
            reply = QMessageBox.question(self, '選択確認', msg_waypoint, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                print(f"User selected Yes for waypoint: {nearest_wp}")
                now_id, now_name, now_i = self.gcl._get_localication_id()
                
                # self.gcl._navigate_route_to_user_waypoint(now_id, now_id, nearest_wp, nearest_wp)
                wp_thread = Thread(target=self.gcl._navigate_route_to_user_waypoint, args=(now_id, now_id, nearest_wp, nearest_wp), daemon=True)
                wp_thread.start()
            else:
                print("User selected No")
        else:
            QToolTip.showText(QCursor.pos(), "No nearby waypoint found.", self)
        # デフォルトハンドラへの転送として OnLeftButtonUp を呼ぶことで状態をリセット
        # obj.GetInteractorStyle().OnLeftButtonUp()
        obj.GetInteractorStyle().OnLeftButtonDown()
        
def set_default_body_control():
    """Set default body control params to current body position"""
    footprint_R_body = geometry.EulerZXY()
    position = geo.Vec3(x=0.0, y=0.0, z=0.0)
    rotation = footprint_R_body.to_quaternion()
    pose = geo.SE3Pose(position=position, rotation=rotation)
    point = trajectory_pb2.SE3TrajectoryPoint(pose=pose)
    traj = trajectory_pb2.SE3Trajectory(points=[point])
    return spot_command_pb2.BodyControlParams(base_offset_rt_footprint=traj)
        
def create_mobility_params(vel_desired=0.5, ang_desired=0.25):
    speed_limit = geo.SE2VelocityLimit(
        max_vel=geo.SE2Velocity(
            linear=geo.Vec2(x=vel_desired, y=vel_desired),
            angular=ang_desired))
    body_control = set_default_body_control()  # 既存の set_default_body_control を利用
    return spot_command_pb2.MobilityParams(vel_limit=speed_limit, obstacle_params=None,
                                             body_control=body_control,
                                             locomotion_hint=spot_command_pb2.HINT_TROT)

def main():
    parser = argparse.ArgumentParser(description="PyQtで表示するGraph Nav Map")
    # bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--path', type=str, help='Map のルートディレクトリパス')
    parser.add_argument('-a', '--anchoring', action='store_true', help='アンカリングに基づいて描画')
    parser.add_argument('--hide-waypoint-text', action='store_true', help='Waypointテキストを非表示')
    parser.add_argument('--hide-world-object-text', action='store_true', help='World Objectテキストを非表示')
    args = parser.parse_args()
    
    
    # bosdyn.client.util.add_payload_credentials_arguments(parser)
    # options = parser.parse_args()
    print("start program")
    
    try:
        sdk = bosdyn.client.create_standard_sdk('SpotFollowClient')
        robot = sdk.create_robot("192.168.80.3")

        recording_sdk = bosdyn.client.create_standard_sdk('SpotRecordingClient')
        print("before authtication")
        # if options.payload_credentials_file:
        #     robot.authenticate_from_payload_credentials(
        #         *bosdyn.client.util.get_guid_and_secret(options))
        # else:
        #     bosdyn.client.util.authenticate(robot)
        bosdyn.client.util.authenticate(robot)
        print("Authetication finish")
        # Time sync is necessary so that time-based filter requests can be converted
        robot.time_sync.wait_for_sync()
        print("time sync")

        # Verify the robot is not estopped and that an external application has registered and holds
        # an estop endpoint.
        assert not robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client,' \
                                        ' such as the estop SDK example, to configure E-Stop.'

        print("estop")
        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        
        client = {
            "state": robot_state_client,
            "command": robot_command_client,
            "lease": lease_client,
        }

        print("create client")

        robot_state_task = AsyncRobotState(robot_state_client)

        print('Detect and follow client connected.')

        lease = lease_client.take()
        lease_keep = LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)
        # Power on the robot and stand it up
        resp = robot.power_on()


        try:
            blocking_stand(robot_command_client)
        except CommandFailedError as exc:
            print(f'Error ({exc}) occurred while trying to stand. Check robot surroundings.')
            return False
        except CommandTimedOutError as exc:
            print(f'Stand command timed out: {exc}')
            return False
        print('Robot powered on and standing.')
        params_set = create_mobility_params()
        
        """graphnavに関する部分"""
        print("GraphNav setting")
        client_metadata = GraphNavRecordingServiceClient.make_client_metadata(
            session_name="", client_username="", client_id='RecordingClient',
            client_type='Python SDK')
        
        app = QApplication(sys.argv)
        window = MainWindow(args.path, args.anchoring, args.hide_waypoint_text, 
                            args.hide_world_object_text, 
                            robot, client)
        window.show()
        sys.exit(app.exec_())
        
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error('Spot jetson threw an exception: %s', exc)
        return False

    

if __name__ == '__main__':
    main()
