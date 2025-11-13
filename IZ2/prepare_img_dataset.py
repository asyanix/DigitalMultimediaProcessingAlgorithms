import cv2
import os
import numpy as np
import random

# Извлекает заданное количество случайных кадров из видео и сохраняет их в указанную директорию.
def extract_random_frames(video_path, num_frames = 150, output_dir = 'res'):
    if not os.path.exists(video_path):
        print(f"Ошибка: Видеофайл не найден по пути: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        print(f"Предупреждение: Количество кадров в видео ({total_frames}) меньше, чем запрошено ({num_frames}). Будут сохранены все кадры.")
        frames_to_extract = total_frames
    else:
        frames_to_extract = num_frames
    
    random_frame_indices = sorted(random.sample(range(total_frames), frames_to_extract))
    
    print(f"Всего кадров в видео: {total_frames}")
    print(f"Извлекается {frames_to_extract} случайных кадров...")
    
    extracted_count = 0

    for i, frame_index in enumerate(random_frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if ret:
            filename = os.path.join(output_dir, f"frame_{i:04d}_{frame_index:06d}.png")
            cv2.imwrite(filename, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            extracted_count += 1
        else:
            print(f"Предупреждение: Не удалось прочитать кадр с индексом {frame_index}")


    cap.release()
    print(f"Готово. Извлечено и сохранено {extracted_count} кадров в папку '{output_dir}'.")
    

if __name__ == "__main__":
    INPUT_VIDEO_PATH = 'IZ2/src/5555.mp4' 
    OUTPUT_FOLDER = 'IZ2/res'
    NUMBER_OF_FRAMES = 150
    
    extract_random_frames(INPUT_VIDEO_PATH, NUMBER_OF_FRAMES, OUTPUT_FOLDER)