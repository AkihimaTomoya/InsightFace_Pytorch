import cv2
from PIL import Image
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import argparse

parser = argparse.ArgumentParser(description='for face verification')
parser.add_argument("-s", "--save", help="whether save", action="store_true")
parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true", default=False)
parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")
parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true", default=True)
args = parser.parse_args()

conf = get_config(False)

mtcnn = MTCNN()
print('mtcnn loaded')

learner = face_learner(conf, True)
learner.threshold = args.threshold
if conf.device.type == 'cpu':
    learner.load_state(conf, 'cpu_final.pth', True, True)
else:
    learner.load_state(conf, 'final.pth', True, True)
learner.model.eval()
print('learner loaded')

if args.update:
    targets, names = prepare_facebank(conf, learner.model, mtcnn, tta=args.tta)
    print('facebank updated')
    print(f'Facebank updated. Found {len(names)} identities.')
else:
    targets, names = load_facebank(conf)
    print('facebank loaded')
    print(f'Facebank loaded. Found {len(names)} identities.')


class faceRec:
    def __init__(self):
        self.width = 800
        self.height = 800
        self.image = None
        self.last_recognition_results = {}  # {face_id: {"name": name, "confidence": conf}

    def get_face_locations(self, frame):
        face_locations = []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            image = Image.fromarray(rgb_frame)
            try:
                res = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)

                if res is not None:
                    bboxes, faces = res

                    #Debug
                    print(
                        f"MTCNN detection: bboxes shape: {bboxes.shape if bboxes is not None else None}, faces length: {len(faces) if faces is not None else None}")

                    if faces is not None and len(faces) > 0 and bboxes is not None and bboxes.size > 0:
                        bboxes = bboxes[:, :-1].astype(int)
                        bboxes = bboxes + [-1, -1, 1, 1]

                        # Chỉ trả về các tọa độ của bounding box
                        for idx, bbox in enumerate(bboxes):
                            face_id = f"face_{idx}"
                            face_locations.append({
                                "id": face_id,
                                "bbox": bbox.tolist()  # Convert numpy array to list [x1, y1, x2, y2]
                            })
            except ValueError as e:
                print(f"MTCNN error: {e} - skipping this frame")
        except Exception as e:
            print(f"Error in get_face_locations: {e}")
            import traceback
            traceback.print_exc()  # In ra stack trace đầy đủ

        return face_locations

    def recognize_faces(self, frame):
        recognition_data = {}

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            image = Image.fromarray(rgb_frame)
            res = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)

            if res is not None:
                bboxes, faces = res

                if faces is not None and len(faces) > 0 and bboxes is not None and bboxes.size > 0:
                    bboxes = bboxes[:, :-1].astype(int)
                    bboxes = bboxes + [-1, -1, 1, 1]
                    results, score = learner.infer(conf, faces, targets, args.tta)

                    # Lưu trữ kết quả nhận dạng cho mỗi khuôn mặt
                    for idx, bbox in enumerate(bboxes):
                        face_id = f"face_{idx}"
                        name = "Unknown"
                        confidence = float('{:.2f}'.format(score[idx]))

                        if confidence >= 0.80:
                            name = names[results[idx] + 1]

                        recognition_data[face_id] = {
                            "name": name,
                            "confidence": confidence
                        }

                    # Cập nhật kết quả nhận dạng
                    self.last_recognition_results = recognition_data
                    print(self.last_recognition_results)
        except Exception as e:
            print(f"Error in recognize_faces: {e}")
            import traceback
            traceback.print_exc()  # In ra stack trace đầy đủ

        return recognition_data
