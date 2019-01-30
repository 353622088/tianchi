# coding:utf-8 
'''
created on 2019/1/29

@author:Dxq
'''
import pandas as pd


def csv_loader():
    df_test = pd.read_csv("train1_label.csv", header=None)
    df_test.columns = ['filename', 'label_name', 'label']

    df_test_length = df_test[
        (df_test.label_name == 'skirt_length_labels') | (df_test.label_name == 'sleeve_length_labels')
        | (df_test.label_name == 'coat_length_labels') | (df_test.label_name == 'pant_length_labels')]

    df_test_design = df_test[
        (df_test.label_name == 'collar_design_labels') | (df_test.label_name == 'lapel_design_labels')
        | (df_test.label_name == 'neckline_design_labels') | (df_test.label_name == 'neck_design_labels')]
    df_test_length.to_csv("length_label.csv", index=False, header=None)
    df_test_design.to_csv("design_label.csv", index=False, header=None)


csv_loader()
