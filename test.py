import os, sys, cv2, csv, argparse, random
import numpy as np
sys.path.append(os.path.abspath(sys.path[0]+'/..'))
from my_code.models.models import build_age_model

def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--data", type=str, default='foo_test.csv', help="Dataset labels")
    parser.add_argument("--images", type=str, default='foo_test/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results.csv', help="CSV file of the results")
    args = parser.parse_args()
    return args

args = init_parameter()
model = build_age_model()
model.load_weights("./my_code/checkpoints/best_last.h5")
# Reading CSV test file
with open(args.data, mode='r') as csv_file:
    gt = csv.reader(csv_file, delimiter=',')
    gt_num = 0
    gt_dict = {}
    for row in gt:
        gt_dict.update({row[0]: int(round(float(row[1])))})
        gt_num += 1
print(gt_num)


# Opening CSV results file
with open(args.results, 'w', newline='') as res_file:
    writer = csv.writer(res_file)
    # Processing all the images
    for image in gt_dict.keys():
        img = cv2.imread(args.images+image)
        if img.size == 0:
            print("Error")
        img = cv2.resize(img, (224,224))
        # Here you should add your code for applying your DCNN
        age = np.argmax(model.predict(np.array([img])/255.))
        # Writing a row in the CSV file
        writer.writerow([image, age])

    

