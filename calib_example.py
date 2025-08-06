import cv2
import numpy as np

# 체커보드 내부 코너 수 (가로, 세로)
CHECKERBOARD = (9, 6)

# 체커보드의 3D 세계 좌표 설정 (Z=0 평면에 있는 정사각형 그리드)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 저장할 포인트 목록
objpoints = []  # 3D 점 (세계 좌표)
imgpoints = []  # 2D 점 (이미지 좌표)

# 이미지 불러오기 (청중이 촬영한 이미지 사용)
img = cv2.imread('data/sample.jpg')  # 이미지 경로를 여기에
img = cv2.resize(img, (1280, 720))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 코너 찾기
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

if ret:
    objpoints.append(objp)
    # 코너 정밀화
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    imgpoints.append(corners2)

    # 코너 시각화
    img_vis = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners2, ret)

    # 카메라 캘리브레이션
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist.ravel())

    # 리프로젝션 예시 이미지 생성
    img_reproj = img.copy()
    imgpoints2, _ = cv2.projectPoints(objp, rvecs[0], tvecs[0], mtx, dist)
    for p in imgpoints2:
        cv2.circle(img_reproj, tuple(p.ravel().astype(int)), 5, (0, 255, 0), -1)

    # 왜곡 보정 (Undistortion)
    print("\n이미지 왜곡 보정 중...")
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    # ROI 크롭
    x, y, w_roi, h_roi = roi
    dst_cropped = dst[y:y+h_roi, x:x+w_roi]
    
    print(f"원본 이미지 크기: {img.shape}")
    print(f"왜곡 보정 후 이미지 크기: {dst_cropped.shape}")
    
    # 결과 이미지 보여주기
    resized_img_vis = cv2.resize(img_vis, (1280, 720))
    resized_img_reproj = cv2.resize(img_reproj, (1280, 720))
    resized_dst = cv2.resize(dst_cropped, (1280, 720))
    
    cv2.imshow('Original Image', cv2.resize(img, (1280, 720)))
    cv2.imshow('Detected Corners', resized_img_vis)
    cv2.imshow('Reprojected Points', resized_img_reproj)
    cv2.imshow('Undistorted Image', resized_dst)
    
    # 왜곡 보정된 이미지 저장
    # cv2.imwrite('data/undistorted_image.jpg', dst_cropped)
    print("왜곡 보정된 이미지가 'undistorted_image.jpg'로 저장되었습니다.")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("체커보드 코너를 찾지 못했습니다. 이미지가 선명한지 확인하세요.")
