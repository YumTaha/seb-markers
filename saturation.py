import os

import cv2
import numpy as np


def change_saturation(image: np.ndarray, scale: float = 1.5) -> np.ndarray:
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
	hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
	return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def main() -> None:
	camera = cv2.VideoCapture(0)
	if not camera.isOpened():
		raise RuntimeError("Unable to access camera")

	ok, frame = camera.read()
	camera.release()

	if not ok:
		raise RuntimeError("Failed to capture image")

	edited = change_saturation(frame, scale=2.5)

	side_by_side = np.concatenate([frame, edited], axis=1)
	output_path = os.path.join(os.getcwd(), "captured_saturation.png")
	cv2.imwrite(output_path, side_by_side)
	print(f"Saved image: {output_path}")


if __name__ == "__main__":
	main()
