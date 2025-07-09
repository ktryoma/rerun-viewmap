import argparse
import os
import sys
import numpy as np
import time
from threading import Thread
import threading

# rerunのインポート
import rerun as rr

# Boston Dynamics SDKのインポート
from bosdyn.api.graph_nav import map_pb2
from bosdyn.client.frame_helpers import *
from bosdyn.client.math_helpers import *

# カラーマップ用の簡易関数
def create_colormap(values, colormap='viridis'):
    """簡易的なカラーマップ関数"""
    if len(values) == 0:
        return np.array([])
    
    normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)
    
    if colormap == 'viridis':
        # viridis風のカラーマップ
        colors = np.zeros((len(values), 3))
        colors[:, 0] = normalized * 68 + 32  # Purple to yellow-green (R)
        colors[:, 1] = normalized * 255  # Green channel
        colors[:, 2] = (1 - normalized) * 255 + normalized * 84  # Blue to yellow (B)
    else:
        # デフォルトの赤-緑グラデーション
        colors = np.zeros((len(values), 3))
        colors[:, 0] = normalized * 255
        colors[:, 1] = (1 - normalized) * 255
        colors[:, 2] = 128
    
    return colors.astype(np.uint8)

def load_map(path):
    with open(os.path.join(path, 'graph'), 'rb') as graph_file:
        # グラフファイルを読み込んでデシリアライズ
        data = graph_file.read()
        current_graph = map_pb2.Graph()
        current_graph.ParseFromString(data)

        # ウェイポイントID、エッジ、スナップショットなどのマップを設定
        current_waypoints = {}
        current_waypoint_snapshots = {}
        current_edge_snapshots = {}
        current_anchors = {}
        current_anchored_world_objects = {}

        # アンカー付きワールドオブジェクトを最初に読み込む
        for anchored_world_object in current_graph.anchoring.objects:
            current_anchored_world_objects[anchored_world_object.id] = (anchored_world_object,)

        # 各ウェイポイントについて、関連するスナップショットを読み込む
        for waypoint in current_graph.waypoints:
            current_waypoints[waypoint.id] = waypoint

            if len(waypoint.snapshot_id) == 0:
                continue
            # スナップショットを読み込む（スナップショットにはウェイポイントのすべての生データが含まれ、大きくなる可能性がある）
            file_name = os.path.join(path, 'waypoint_snapshots', waypoint.snapshot_id)
            if not os.path.exists(file_name):
                continue
            with open(file_name, 'rb') as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                try:
                    waypoint_snapshot.ParseFromString(snapshot_file.read())
                    current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
                except Exception as e:
                    print(f"ウェイポイントスナップショット {waypoint.snapshot_id} の読み込みに失敗: {e}")

                # アンカー付きワールドオブジェクトのfiducialを確認
                for fiducial in waypoint_snapshot.objects:
                    if fiducial.id in current_anchored_world_objects:
                        # タプルを拡張して (anchored_world_object, waypoint, fiducial) にする
                        current_anchored_world_objects[fiducial.id] = (
                            current_anchored_world_objects[fiducial.id][0], waypoint, fiducial)

        # エッジにもスナップショットデータがある
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
            f'グラフを読み込みました: ウェイポイント{len(current_graph.waypoints)}個, エッジ{len(current_graph.edges)}個, '
            f'アンカー{len(current_graph.anchoring.anchors)}個, アンカー付きワールドオブジェクト{len(current_graph.anchoring.objects)}個'
        )
        return (current_graph, current_waypoints, current_waypoint_snapshots,
                current_edge_snapshots, current_anchors, current_anchored_world_objects)

def create_point_cloud_data(waypoints, snapshots, waypoint_id):
    wp = waypoints[waypoint_id]
    snapshot = snapshots[wp.snapshot_id]
    cloud = snapshot.point_cloud
    
    try:
        odom_tform_cloud = get_a_tform_b(cloud.source.transforms_snapshot, ODOM_FRAME_NAME,
                                         cloud.source.frame_name_sensor)
        
        if odom_tform_cloud is None:
            print(f"ウェイポイント {waypoint_id} の点群変換行列を取得できませんでした")
            return np.empty((0, 3))
            
        waypoint_tform_odom = SE3Pose.from_proto(wp.waypoint_tform_ko)
        waypoint_tform_cloud = waypoint_tform_odom * odom_tform_cloud

        point_cloud_data = np.frombuffer(cloud.data, dtype=np.float32).reshape(int(cloud.num_points), 3)
        
        # 点群データをワールド座標系に変換
        transform_matrix = waypoint_tform_cloud.to_matrix()
        homogeneous_points = np.hstack([point_cloud_data, np.ones((point_cloud_data.shape[0], 1))])
        transformed_points = (transform_matrix @ homogeneous_points.T).T[:, :3]
        
        return transformed_points
    except Exception as e:
        print(f"ウェイポイント {waypoint_id} の点群処理でエラー: {e}")
        return np.empty((0, 3))

def visualize_anchored_graph(current_graph, current_waypoint_snapshots, current_waypoints,
                           current_anchors, current_anchored_world_objects, current_waypoint_id=None):
    waypoint_positions = {}

    # ウェイポイントを可視化
    waypoint_points = []
    waypoint_colors = []
    waypoint_labels = []

    for waypoint in current_graph.waypoints:
        if waypoint.id in current_anchors:
            seed_tform_waypoint = SE3Pose.from_proto(
                current_anchors[waypoint.id].seed_tform_waypoint).to_matrix()
            position = seed_tform_waypoint[:3, 3]
            waypoint_positions[waypoint.id] = position

            waypoint_points.append(position)
            waypoint_colors.append([0, 100, 255])  # 青色（現在地の赤色と区別）

            # ウェイポイント名を追加
            name = waypoint.annotations.name if waypoint.annotations.name else waypoint.id
            waypoint_labels.append(name)

            # 点群データがある場合は可視化
            if waypoint.snapshot_id in current_waypoint_snapshots:
                try:
                    point_cloud_data = create_point_cloud_data(current_waypoints, 
                                                             current_waypoint_snapshots, 
                                                             waypoint.id)
                    if len(point_cloud_data) > 0:
                        # 点群をseed frameに変換
                        homogeneous_points = np.hstack([point_cloud_data, np.ones((point_cloud_data.shape[0], 1))])
                        transformed_points = (seed_tform_waypoint @ homogeneous_points.T).T[:, :3]

                        # Z座標に基づく色付け
                        z_coords = transformed_points[:, 2]
                        colors = create_colormap(z_coords, 'viridis')

                        waypoint_points.extend(transformed_points)
                        waypoint_colors.extend(colors)
                except Exception as e:
                    print(f"ウェイポイント {waypoint.id} の点群作成に失敗: {e}")

    # ウェイポイントを点として描画
    if waypoint_points:
        rr.log("waypoints", 
              rr.Points3D(waypoint_points, colors=waypoint_colors, radii=0.1, labels=waypoint_labels))

    # エッジを可視化
    edge_lines = []
    for edge in current_graph.edges:
        if edge.id.from_waypoint in current_anchors and edge.id.to_waypoint in current_anchors:
            from_pos = waypoint_positions[edge.id.from_waypoint]
            to_pos = waypoint_positions[edge.id.to_waypoint]
            edge_lines.append([from_pos, to_pos])

    if edge_lines:
        rr.log("edges", 
              rr.LineStrips3D(edge_lines, colors=[0, 255, 0]))

    # アンカー付きワールドオブジェクトを可視化
    world_object_points = []
    world_object_colors = []
    world_object_labels = []

    for anchored_wo in current_anchored_world_objects.values():
        if len(anchored_wo) >= 1:  # anchored_world_objectが存在する場合
            anchored_world_object = anchored_wo[0]
            seed_tform_object = SE3Pose.from_proto(anchored_world_object.seed_tform_object).to_matrix()
            position = seed_tform_object[:3, 3]

            world_object_points.append(position)
            world_object_colors.append([255, 0, 255])  # マゼンタ色
            world_object_labels.append(anchored_world_object.id)

    if world_object_points:
        rr.log("world_objects", 
              rr.Points3D(world_object_points, colors=world_object_colors, radii=0.15, labels=world_object_labels))

def visualize_graph(current_graph, current_waypoint_snapshots, current_waypoints, current_waypoint_id=None):
    waypoint_positions = {}
    
    # 幅優先探索でグラフを構築
    queue = []
    queue.append((current_graph.waypoints[0], np.eye(4)))
    visited = {}
    
    waypoint_points = []
    waypoint_colors = []
    waypoint_labels = []
    
    while len(queue) > 0:
        curr_element = queue.pop(0)
        curr_waypoint = curr_element[0]
        
        if curr_waypoint.id in visited:
            continue
        visited[curr_waypoint.id] = True
        
        # 現在のウェイポイントのワールド座標を設定
        world_tform_current_waypoint = curr_element[1]
        position = world_tform_current_waypoint[:3, 3]
        waypoint_positions[curr_waypoint.id] = position
        
        waypoint_points.append(position)
        waypoint_colors.append([0, 100, 255])  # 青色（現在地の赤色と区別）
        
        # ウェイポイント名を追加
        name = curr_waypoint.annotations.name if curr_waypoint.annotations.name else curr_waypoint.id
        waypoint_labels.append(name)
        
        # 点群データがある場合は可視化
        if curr_waypoint.snapshot_id in current_waypoint_snapshots:
            try:
                point_cloud_data = create_point_cloud_data(current_waypoints, 
                                                         current_waypoint_snapshots, 
                                                         curr_waypoint.id)
                if len(point_cloud_data) > 0:
                    # 点群をワールド座標系に変換
                    homogeneous_points = np.hstack([point_cloud_data, np.ones((point_cloud_data.shape[0], 1))])
                    transformed_points = (world_tform_current_waypoint @ homogeneous_points.T).T[:, :3]
                    
                    # Z座標に基づく色付け
                    z_coords = transformed_points[:, 2]
                    colors = create_colormap(z_coords, 'viridis')
                    
                    rr.log(f"waypoints/pointcloud/{curr_waypoint.id}", 
                          rr.Points3D(transformed_points, colors=colors, radii=0.02))
            except Exception as e:
                print(f"ウェイポイント {curr_waypoint.id} の点群作成に失敗: {e}")
        
        # fiducialオブジェクトを可視化
        if curr_waypoint.snapshot_id in current_waypoint_snapshots:
            snapshot = current_waypoint_snapshots[curr_waypoint.snapshot_id]
            for fiducial in snapshot.objects:
                if fiducial.apriltag_properties:
                    try:
                        # fiducialの位置を計算
                        odom_tform_fiducial = get_a_tform_b(
                            fiducial.transforms_snapshot, ODOM_FRAME_NAME,
                            fiducial.apriltag_properties.frame_name_fiducial_filtered)
                        
                        if odom_tform_fiducial is None:
                            # print(f"fiducial {fiducial.id} の変換行列を取得できませんでした")
                            continue
                            
                        waypoint_tform_odom = SE3Pose.from_proto(curr_waypoint.waypoint_tform_ko)
                        waypoint_tform_fiducial = waypoint_tform_odom * odom_tform_fiducial
                        world_tform_fiducial = world_tform_current_waypoint @ waypoint_tform_fiducial.to_matrix()
                        
                        fiducial_pos = world_tform_fiducial[:3, 3]
                        rr.log(f"fiducials/{fiducial.id}", 
                              rr.Points3D([fiducial_pos], colors=[0, 0, 255], radii=0.1))
                        
                        # テキストを3D空間に配置
                        text_position = fiducial_pos + np.array([0, 0, 0.5])
                        rr.log(f"fiducials/text/{fiducial.id}", 
                              rr.Transform3D(translation=text_position))
                        rr.log(f"fiducials/text/{fiducial.id}", 
                              rr.TextDocument(text=fiducial.id))
                    except Exception as e:
                        print(f"fiducial {fiducial.id} の処理に失敗: {e}")
                        continue
        
        # エッジを探索してキューに追加
        for edge in current_graph.edges:
            if edge.id.from_waypoint == curr_waypoint.id and edge.id.to_waypoint not in visited:
                # エッジ変換を連結
                curr_wp_tform_to_wp = SE3Pose.from_proto(edge.from_tform_to).to_matrix()
                world_tform_to_wp = np.dot(world_tform_current_waypoint, curr_wp_tform_to_wp)
                to_waypoint = current_waypoints[edge.id.to_waypoint]
                queue.append((to_waypoint, world_tform_to_wp))
                
            elif edge.id.to_waypoint == curr_waypoint.id and edge.id.from_waypoint not in visited:
                # 逆方向のエッジ
                to_wp_tform_curr_wp = SE3Pose.from_proto(edge.from_tform_to).inverse().to_matrix()
                world_tform_from_wp = np.dot(world_tform_current_waypoint, to_wp_tform_curr_wp)
                from_waypoint = current_waypoints[edge.id.from_waypoint]
                queue.append((from_waypoint, world_tform_from_wp))

    # ウェイポイントを点として描画
    if waypoint_points:
        # print(waypoint_points)
        rr.log("waypoints/positions", 
              rr.Points3D(waypoint_points, colors=waypoint_colors, radii=0.5))
        
    # 任意のwaypoint.idの座標を探し出してposに格納
    target_waypoint_id = "amiss-gaur-DOLhLXzIIYWkjGlktRDifA=="  # 探したいウェイポイントID
    pos = None
    
    if target_waypoint_id in waypoint_positions:
        pos = waypoint_positions[target_waypoint_id]
        print(f"ウェイポイント {target_waypoint_id} の座標: {pos}")
        
        # 見つかった座標を赤色で描画
        rr.log("waypoints/positions/target", 
              rr.Points3D([pos], colors=[[255, 0, 0]], radii=1.5))
    else:
        print(f"ウェイポイント {target_waypoint_id} が見つかりません")
        print(f"利用可能なウェイポイントID: {list(waypoint_positions.keys())}")
        

    # 現在地を表示
    # display_current_location(current_graph, waypoint_positions, None, current_waypoint_id)

    # エッジを可視化
    edge_lines = []
    for edge in current_graph.edges:
        if edge.id.from_waypoint in waypoint_positions and edge.id.to_waypoint in waypoint_positions:
            from_pos = waypoint_positions[edge.id.from_waypoint]
            to_pos = waypoint_positions[edge.id.to_waypoint]
            edge_lines.append([from_pos, to_pos])
    
    if edge_lines:
        rr.log("edges", 
              rr.LineStrips3D(edge_lines, colors=[0, 255, 0]))
    
    # waypoint_positionsを返す
    return waypoint_positions

def update_current_location_thread(current_graph, waypoint_positions, current_anchors=None):
    """現在位置を定期的に更新するスレッド関数"""
    while True:
        try:
            # ここで実際のロボットから現在位置を取得する処理を追加
            # 例: current_waypoint_id = get_robot_current_waypoint()
            # 今は例として最初のウェイポイントを使用
            current_waypoint_id = list(waypoint_positions.keys())[0] if waypoint_positions else None
            
            if current_waypoint_id:
                display_current_location(current_graph, waypoint_positions, current_anchors, current_waypoint_id)
            
            time.sleep(1)  # 1秒待機
        except Exception as e:
            print(f"現在位置更新エラー: {e}")
            time.sleep(1)

def display_current_location(current_graph, waypoint_positions, current_anchors=None, current_waypoint_id=None):
    # 現在地のウェイポイントIDを決定
    if current_waypoint_id is None:
        if len(current_graph.waypoints) > 0:
            current_waypoint_id = current_graph.waypoints[0].id
        else:
            return
    
    # 現在地の位置を取得
    current_position = None
    
    if current_anchors is not None:
        # アンカーモードの場合
        if current_waypoint_id in current_anchors:
            seed_tform_waypoint = SE3Pose.from_proto(
                current_anchors[current_waypoint_id].seed_tform_waypoint).to_matrix()
            current_position = seed_tform_waypoint[:3, 3]
    else:
        # 通常モードの場合
        if current_waypoint_id in waypoint_positions:
            current_position = waypoint_positions[current_waypoint_id]
    
    if current_position is not None:
        # 大きな赤い点で現在地マーカーを表示
        rr.log("current_location", 
              rr.Points3D([current_position], colors=[[255, 255, 0]], radii=0.8))
        
        # 現在地のテキストラベル（Z座標で1上に表示）
        text_position = current_position + np.array([0, 0, 1.0])
        rr.log("current_location/text", 
              rr.Transform3D(translation=text_position))
        rr.log("current_location/text", 
              rr.TextDocument(text="現在地"))
        
        print(f"現在地を更新: {current_waypoint_id} at {current_position}")
    else:
        print(f"警告: 指定されたウェイポイント '{current_waypoint_id}' が見つかりません")

def print_waypoint_info(waypoint):
    print("--- Waypoint Information ---")
    print(f"ID: {waypoint.id}")
    if waypoint.annotations.name:
        print(f"Name: {waypoint.annotations.name}")
    else:
        print("Name: None")
    print(f"Snapshot ID: {waypoint.snapshot_id if waypoint.snapshot_id else 'None'}")
    print(f"Annotations: {waypoint.annotations}")
    print("----------------------------")

def main():
    parser = argparse.ArgumentParser(description="visualize Boston Dynamics GraphNav map using rerun")
    parser.add_argument('--path', type=str, default='downloaded_graph', help='root directory of the map')
    parser.add_argument('-a', '--anchoring', action='store_true',
                        help='draw based on anchoring (seed frame)')
    args = parser.parse_args()

    # Check if map files exist
    if not os.path.exists(args.path):
        print(f"エラー: マップディレクトリ '{args.path}' が見つかりません")
        return

    # rerunを初期化
    rr.init("GraphNav Map Viewer", spawn=True)

    try:
        # マップを読み込み
        (current_graph, current_waypoints, current_waypoint_snapshots,
         current_edge_snapshots, current_anchors, current_anchored_world_objects) = load_map(args.path)

        # Waypoint[0]の情報を表示
        if current_graph.waypoints:
            print_waypoint_info(current_graph.waypoints[0])

        # 可視化
        waypoint_positions = {}
        if args.anchoring:
            if len(current_graph.anchoring.anchors) == 0:
                print('警告: 描画するアンカーがありません。通常モードで描画します。')
                visualize_graph(current_graph, current_waypoint_snapshots, current_waypoints)
            else:
                visualize_anchored_graph(current_graph, current_waypoint_snapshots, current_waypoints,
                                       current_anchors, current_anchored_world_objects)
                # アンカーモードでのwaypoint_positionsを取得
                for waypoint in current_graph.waypoints:
                    if waypoint.id in current_anchors:
                        seed_tform_waypoint = SE3Pose.from_proto(
                            current_anchors[waypoint.id].seed_tform_waypoint).to_matrix()
                        waypoint_positions[waypoint.id] = seed_tform_waypoint[:3, 3]
        else:
            # 通常モードでのwaypoint_positionsを取得するため、visualize_graphを修正する必要があります
            waypoint_positions = visualize_graph(current_graph, current_waypoint_snapshots, current_waypoints)

        # 現在位置更新スレッドを開始
        location_thread = threading.Thread(
            target=update_current_location_thread, 
            args=(current_graph, waypoint_positions, current_anchors if args.anchoring else None),
            daemon=True
        )
        location_thread.start()

        print("rerun viewerでマップを確認してください。")
        print("現在位置は1秒ごとに更新されます。")
        print("プログラムを終了するには Ctrl+C を押してください。")

        # プログラムを実行し続ける
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nプログラムを終了します。")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
