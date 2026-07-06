"""XFeat params — defaults from accelerated_features/realtime_demo.py."""

XFEAT_TOP_K = 3000
PEG_HOMOGRAPHY_MIN_INLIERS = 10
HOMOGRAPHY_RANSAC = 4.0
MATCH_COSSIM = 0.82
ZONE_MARGIN = 48
ZONE_SEARCH = 260
PEG_SLIDE_STEP = 24
# coarse：周期性 XFeat 重匹配；中间步保留上一帧 XFeat 角点（无 GT/几何补帧）
MATCH_EVERY = 16
MATCH_EVERY_GUI = 8
COARSE_MAX_PROVIDER_CALLS = 720
# 单角点相对示教的最大像素位移（拒 ep7 级 wild warp）
SANITY_MAX_CORNER_SHIFT_PX = 280
