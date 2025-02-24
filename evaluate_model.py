from config import get_config
import argparse
from Learner import face_learner
from data.data_pipe import get_val_pair
from torchvision import transforms as trans
import os

def main():
    parser = argparse.ArgumentParser(description="Face Recognition Evaluation")
    parser.add_argument("-mb", action="store_true", help="Use MobileFaceNet model (default: IR_SE50)")

    args = parser.parse_args()
  
    conf = get_config(False)
  
    if args.mb:
        model_path = "mobilefacenet.pth"
        conf.use_mobilfacenet = True
    else:
        model_path = "ir_se50.pth"
      
    learner = face_learner(conf, inference=True)
    learner.load_state(conf, model_path, model_only=True, from_save_folder=False)

    datasets = ['agedb_30', 'calfw', 'cfp_ff', 'cfp_fp', 'cplfw', 'lfw']
    
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    for dataset in datasets:
        data, data_issame = get_val_pair(conf.webface_folder, dataset)
        accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, data, data_issame, nrof_folds=10, tta=True)
        
        print(f'{dataset} - Accuracy: {accuracy}, Threshold: {best_threshold}')
        
        image = trans.ToPILImage()(roc_curve_tensor)
        image_path = os.path.join(output_folder, f"{dataset}_roc_curve.png")
        image.save(image_path)
        print(f"Saved ROC curve for {dataset} at {image_path}")

if __name__ == "__main__":
    main()
