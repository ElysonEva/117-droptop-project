This Folder will hold the video and dataset paths to replicate code for Windows 10.
Implemented Set Logic. Each Course segment will hold a set that holds the relevant Droplets in that designated course. As well as the next segment.
Imagine 2 pointers that as they traverse the path hold the information of the relevant Droplets at any given point in time. Set logic improves average runtime.
Another method utilized is a set found logic. Such that if a droplet is associated to a detection in one round it won't be rechecked by another detection. Similar to Dijkstra's Visited Set logic.

ds_with_set_logic.py is the version implementing only set logic
ds_with_set_logic_with_curve_infer.py implements a prediction model for where a droplet will be in the curve to help improve accuracy
