import os
import sys
import subprocess
import cv2
import argparse
from tqdm import tqdm

def download_bilibili_video(url, output_dir):
    """
    使用you-get下载B站视频
    
    Args:
        url: B站视频链接
        output_dir: 视频保存目录
    
    Returns:
        下载的视频文件路径
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 使用subprocess调用you-get下载视频
    print(f"正在下载视频: {url}")
    cmd = ["you-get", "-o", output_dir, url]
    
    try:
        subprocess.run(cmd, check=True)
        # 获取下载的视频文件名
        video_files = [f for f in os.listdir(output_dir) if f.endswith(('.mp4', '.flv'))]
        if video_files:
            return os.path.join(output_dir, video_files[0])
        else:
            print("未找到下载的视频文件")
            return None
    except subprocess.CalledProcessError as e:
        print(f"下载视频时出错: {e}")
        return None

def extract_frames(video_path, output_dir, frame_interval=1):
    """
    从视频中抽取帧并保存为图片
    
    Args:
        video_path: 视频文件路径
        output_dir: 图片保存目录
        frame_interval: 抽帧间隔（每隔多少帧提取一帧）
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"视频总帧数: {total_frames}, FPS: {fps}")
    print(f"开始抽帧，间隔为 {frame_interval} 帧")
    
    # 抽取帧
    frame_count = 0
    saved_count = 0
    
    # 使用tqdm显示进度条
    with tqdm(total=total_frames) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按照指定间隔抽取帧
            if frame_count % frame_interval == 0:
                # 保存图片
                frame_filename = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    print(f"抽帧完成，共保存 {saved_count} 张图片到 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="下载B站视频并抽帧")
    parser.add_argument("--url", default="https://www.bilibili.com/video/BV1Bm421s7zJ/",help="B站视频URL")
    parser.add_argument("--video_dir", default="videos", help="视频保存目录")
    parser.add_argument("--frame_dir", default="./frames/norm", help="帧图片保存目录")
    parser.add_argument("--interval", type=int, default=300, help="抽帧间隔，默认每30帧抽取一帧")
    
    args = parser.parse_args()
    
    # 下载视频
    video_path = download_bilibili_video(args.url, args.video_dir)
    
    if video_path:
        # 抽取帧
        extract_frames(video_path, args.frame_dir, args.interval)
    else:
        print("视频下载失败，无法进行抽帧")

if __name__ == "__main__":
    main()