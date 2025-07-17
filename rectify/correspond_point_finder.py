import cv2
import numpy as np

def main():
    # Load images
    frame1 = cv2.imread('./rectify/IMG_for_corresponing_point_left.jpeg')
    frame2 = cv2.imread('./rectify/IMG_for_corresponing_point_right.jpeg')

    if frame1 is None or frame2 is None:
        print("이미지 로드 실패")
        return

    sift = cv2.SIFT_create()
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    saved_points = []
    current_match_index = 0

    print("스페이스바: 대응점 찾기 | 좌/우 방향키: 매칭 이동 | r: 좌표 저장 | q: 종료")

    # Perform feature matching once
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        print("특징점이 부족합니다.")
        return

    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good_matches) < 8:
        print(f"매칭이 부족합니다. (매칭 수: {len(good_matches)})")
        return

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # RANSAC으로 Fundamental Matrix 계산 및 필터링
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    if F is None or mask is None:
        print("Fundamental Matrix 계산 실패")
        return

    inlier_pts1 = pts1[mask.ravel() == 1]
    inlier_pts2 = pts2[mask.ravel() == 1]
    inlier_matches = [good_matches[i] for i in range(len(mask)) if mask[i]]

    print(f"총 매칭 수: {len(good_matches)}, Inliers: {len(inlier_pts1)}")

    # Show initial images
    cv2.imshow("Left Image", frame1)
    cv2.imshow("Right Image", frame2)

    # Show first matching pair
    if inlier_matches:
        single_match = [inlier_matches[current_match_index]]
        match_img = cv2.drawMatches(frame1, kp1, frame2, kp2, single_match, None, flags=2)
        cv2.imshow("Current Match", match_img)

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord(' '):  # 스페이스바: 재표시
            if inlier_matches:
                single_match = [inlier_matches[current_match_index]]
                match_img = cv2.drawMatches(frame1, kp1, frame2, kp2, single_match, None, flags=2)
                cv2.imshow("Current Match", match_img)

        elif key == ord('r') and inlier_matches:  # r 키: 좌표 저장
            pt1 = kp1[inlier_matches[current_match_index].queryIdx].pt
            pt2 = kp2[inlier_matches[current_match_index].trainIdx].pt
            saved_points.append((pt1, pt2))
            
            # 파일에 저장
            with open('correspond_point.txt', 'a') as f:
                f.write(f"Point {len(saved_points)}: Left({pt1[0]:.2f}, {pt1[1]:.2f}), Right({pt2[0]:.2f}, {pt2[1]:.2f})\n")
            
            print(f"저장된 점: Left({pt1[0]:.2f}, {pt1[1]:.2f}), Right({pt2[0]:.2f}, {pt2[1]:.2f})")
            
            # 8개 저장 시 종료
            if len(saved_points) >= 8:
                print("8개의 대응점이 저장되었습니다. 프로그램 종료.")
                break

        elif key == 81 and inlier_matches:  # 왼쪽 방향키
            current_match_index = max(0, current_match_index - 1)
            single_match = [inlier_matches[current_match_index]]
            match_img = cv2.drawMatches(frame1, kp1, frame2, kp2, single_match, None, flags=2)
            cv2.imshow("Current Match", match_img)

        elif key == 83 and inlier_matches:  # 오른쪽 방향키
            current_match_index = min(len(inlier_matches) - 1, current_match_index + 1)
            single_match = [inlier_matches[current_match_index]]
            match_img = cv2.drawMatches(frame1, kp1, frame2, kp2, single_match, None, flags=2)
            cv2.imshow("Current Match", match_img)

        elif key == ord('q'):  # q 키: 종료
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()