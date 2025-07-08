# rerun-viewmap
This is a map viewer for Spot robot using [Rerun](https://rerun.io/).  
![Rerun Map Viewer](./fig/overview.png)

## Build environment
Install the following packages:
```bash
pip install bosdyn-client
pip install rerun-sdk
```

## Run
```bash
python rerun_map_viewer.py --path <path_to_map_dir>
```

## History
- 2025/07/08
    - Initial commit
    - Add `rerun_map_viewer.py`

## Future work
- [ ] Add hover information in map viewer
- [ ] Control Spot using map viewer (implement `control.py`)