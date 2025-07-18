label_mapping = {
    '00000000': [0],   # C1 Normal
    '10000000': [1],   # C2 Center
    '01000000': [2],   # C3 Donut
    '00100000': [3],   # C4 Edge_Loc(EL)
    '00010000': [4],   # C5 Edge_Ring(ER)
    '00001000': [5],   # C6 Loc
    '00000100': [6],   # C7 Near_Full(NF)
    '00000010': [7],   # C8 Scratch(S)
    '00000001': [8],   # C9 Random(R)
    '10100000': [9],   # C10 C+EL
    '10010000': [10],  # C11 C+ER
    '10001000': [11],  # C12 C+L
    '10000010': [12],  # C13 C+S
    '01100000': [13],  # C14 D+EL
    '01010000': [14],  # C15 D+ER
    '01001000': [15],  # C16 D+L
    '01000010': [16],  # C17 D+S
    '00101000': [17],  # C18 EL+L
    '00100010': [18],  # C19 EL+S
    '00011000': [19],  # C20 ER+L
    '00010010': [20],  # C21 ER+S
    '00001010': [21],  # C22 L+S
    '10101000': [22],  # C23 C+EL+L
    '10100010': [23],  # C24 C+EL+S
    '10011000': [24],  # C25 C+ER+L
    '10010010': [25],  # C26 C+ER+S
    '10001010': [26],  # C27 C+L+S
    '01101000': [27],  # C28 D+EL+L
    '01100010': [28],  # C29 D+EL+S
    '01011000': [29],  # C30 D+ER+L
    '01010010': [30],  # C31 D+ER+S
    '01001010': [31],  # C32 D+L+S
    '00101010': [32],  # C33 EL+L+S
    '00011010': [33],  # C34 ER+L+S
    '10101010': [34],  # C35 C+L+EL+S
    '10011010': [35],  # C36 C+L+ER+S
    '01101010': [36],  # C37 D+L+EL+S
    '01011010': [37],  # C38 D+L+ER+S

}
# classes = ('Normal', 'Center', 'Donut', 'Edge_Loc', 'Edge_Ring', 'Loc', 'Near_Full', 'Scratch', 'Random', 'C+EL',
#            'C+ER', 'C+L', 'C+S', 'D+EL', 'D+ER', 'D+L', 'D+S', 'EL+L', 'EL+S', 'ER+L', 'ER+S', 'L+S',
#            'C+EL+L', 'C+EL+S', 'C+ER+L', 'C+ER+S', 'C+L+S', 'D+EL+L', 'D+EL+S', 'D+ER+L', 'D+ER+S',
#            'D+L+S', 'EL+L+S', 'ER+L+S', 'C+L+EL+S', 'C+L+ER+S', 'D+L+EL+S', 'D+L+ER+S')
# import numpy as np


# def convert_labels(labels, label_mapping):
#     indices = []
#     for sample in labels:
#         label_str = '[' + ','.join(map(str, sample.astype(int))) + ']'
#         if label_str in label_mapping:
#             indices.append(label_mapping[label_str][0])
#         else:
#             raise ValueError(f"{label_str}not in label_mapping")
#     return np.array(indices)


# if __name__ == '__main__':
#     from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#     y_true = np.array([
#         [0, 0, 0, 0, 0, 0, 0, 0],  # C1
#         [1, 0, 0, 0, 0, 0, 0, 0],  # C2
#         [1, 0, 0, 1, 0, 0, 0, 0],  # C11
#         [1, 0, 0, 1, 1, 0, 1, 0],  # C36
#         [0, 0, 0, 0, 1, 0, 1, 0],  # C22
#         [0, 1, 0, 1, 1, 0, 1, 0],  # C38
#     ])

#     y_pred = np.array([
#         [0, 0, 0, 0, 0, 0, 0, 0],  # C1
#         [1, 0, 0, 0, 0, 0, 0, 0],  # C2
#         [1, 0, 0, 0, 0, 0, 0, 0],  # C2
#         [1, 0, 0, 1, 1, 0, 0, 0],  # C24
#         [0, 0, 0, 0, 1, 0, 1, 0],  # C22
#         [0, 1, 0, 1, 1, 0, 1, 0],  # C38
#     ])


#     y_true_idx = convert_labels(y_true, label_mapping)
#     y_pred_idx = convert_labels(y_pred, label_mapping)


#     accuracy = accuracy_score(y_true_idx, y_pred_idx)
#     precision = precision_score(y_true_idx, y_pred_idx, average='macro')
#     recall = recall_score(y_true_idx, y_pred_idx, average='macro')
#     f1 = f1_score(y_true_idx, y_pred_idx, average='macro')

#     print("true index:", y_true_idx)
#     print("pred index:", y_pred_idx)
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 Score: {f1:.4f}")

