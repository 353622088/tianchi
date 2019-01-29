# coding:utf-8 
"""
created on 2019/1/29

@author:Dxq
"""
import csv
import scipy.io as scio

with open("train2_label.csv") as train_csv:
    csv_reader_lines = csv.reader(train_csv)
    img_types = []
    img_labels = dict()
    for one_line in csv_reader_lines:
        img_path, img_type, img_label = one_line
        img_types.append(img_type)
        if img_type not in img_labels.keys():
            img_labels.update({img_type: []})
        img_labels[img_type].append(img_label)

design_subclass_dict = {
    "collar_design_labels": {"index2label": ["Invisible", "Shirt", "Peter_Pan",
                                             "Puritan", "Rib"],
                             # "Invisible": 1741, "Shirt": 1995,
                             # "Peter Pan": 1526, "Puritan": 1321,
                             # "Rib": 1810
                             },
    "neckline_design_labels": {"index2label": ["Invisible", "Strapless_Neck", "Deep_V_Neckline",
                                               "Straight_Neck", "V_Neckline", "Square_Neckline",
                                               "Off_Shoulder", "Round_Neckline", "Sweat_Heart_Neck",
                                               "One	Shoulder_Neckline"],
                               # "Invisible": 2106, "Strapless Neck": 1926,
                               # "Deep V Neckline": 2472, "Straight Neck": 1105,
                               # "V Neckline": 1897, "Square Neckline": 1625,
                               # "Off Shoulder": 1706, "Round Neckline": 1376,
                               # "Sweat Heart Neck": 1634, "One	Shoulder Neckline": 1301
                               },

    "lapel_design_labels": {"index2label": ["Invisible", "Notched", "Collarless",
                                            "Shawl", "Plus_Size_Shawl"],
                            # "Invisible": 1174, "Notched": 1525,
                            # "Collarless": 1887, "Shawl": 940,
                            # "Plus Size Shawl": 1508
                            },
    "neck_design_labels": {"index2label": ["Invisible", "Turtle_Neck", "Ruffle_Semi-High_Collar",
                                           "Low_Turtle_Neck", "Draped_Collar"],
                           # "Invisible": 802, "Turtle Neck": 1304,
                           # "Ruffle Semi-High Collar": 1192, "Low Turtle Neck": 1454,
                           # "Draped Collar": 944
                           }
}

length_subclass_dict = {
    "coat_length_labels": {"index2label": ["Invisible", "High_Waist", "Regular",
                                           "Long", "Micro", "Knee", "Midi", "Ankle&Floor"],
                           # "Invisible": 715, "High Waist": 1282,
                           # "Regular": 1484, "Long": 984,
                           # "Micro": 1416, "Knee": 2309,
                           # "Midi": 1321, "Ankle&Floor": 1809
                           },
    "skirt_length_labels": {"index2label": ["Invisible", "Short", "Knee",
                                            "Midi", "Ankle", "Floor"],
                            # "Invisible": 1483, "Short": 1372,
                            # "Knee": 1503, "Midi": 1387,
                            # "Ankle": 1495, "Floor": 1983
                            },
    "sleeve_length_labels": {"index2label": ["Invisible", "Sleeveless", "Cup_Sleeves",
                                             "Short_Sleeves", "Elbow_Sleeves", "3of4_Sleeves",
                                             "Wrist_Length", "Long_Sleeves", "Extra_Long_Sleeves"],
                             # "Invisible": 0, "Sleeveless": 1443,
                             # "Cup Sleeves": 2843, "Short Sleeves": 1498,
                             # "Elbow Sleeves": 1311, "3/4 Sleeves": 1633,
                             # "Wrist Length": 1156, "Long Sleeves": 1518,
                             # "Extra Long Sleeves": 1897
                             },
    "pant_length_labels": {"index2label": ["Invisible", "Short_Pant", "Mid_Length",
                                           "3of4_Length", "Cropped_Pant", "Full_Length"],
                           # "Invisible": 1519, "Short Pant": 1348,
                           # "Mid Length": 1087, "3/4 Length": 1321,
                           # "Cropped Pant": 948, "Full Length": 1237
                           }
}

for key in img_labels.keys():
    subclass_dict = design_subclass_dict if "design" in key else length_subclass_dict
    subclass = subclass_dict[key]
    for label in img_labels[key]:
        for i, sub_key in enumerate(subclass["index2label"]):
            if sub_key not in subclass.keys():
                subclass.update({sub_key: 0})
            if label[i] == "y":
                subclass[sub_key] += 1
scio.savemat("design_train2.mat", design_subclass_dict)
scio.savemat("length_train2.mat", length_subclass_dict)
print(design_subclass_dict)
print(length_subclass_dict)
