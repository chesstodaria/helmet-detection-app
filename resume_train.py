from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"D:\2. coolyeah\semester 5\tekber\aplikasi pendeteksi helm\helmet-detection-app\runs\detect\runs\helmet\finetune_v1\weights\last.pt")
    model.train(resume=True, device=0)
