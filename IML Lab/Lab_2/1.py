import matplotlib.pyplot as mp 

actual = [1,0,0,0,1,1,0,0,1,1]
predicted = [1,1,0,1,0,1,1,1,0,1]
n = len(actual)
TP = 0
TN = 0
FP = 0
FN = 0
for i in range(n):
    if(actual[i] == 1 and predicted[i] == 1):
        TP = TP + 1
    elif(actual[i] == 0 and predicted[i] == 0):
        TN = TN + 1
    elif(actual[i] == 0 and predicted[i] == 1):
        FP = FP + 1
    elif(actual[i] == 1 and predicted[i] == 0):
        FN = FN + 1


# precision
precision = 0

if(TP+FP == 0):
    precision = 0
else:
    precision = TP/(TP + FP)

# recall
recall = 0
if(TP+FN == 0):
    recall = 0
else:
    recall = TP/(TP+FN)

# F1
F1 = 0
if(precision + recall == 0):
    F1 = 0
else:
    F1 = (2*precision * recall)/(precision + recall)

# accuracy
accuracy = 0
if(TP + TN + FP + FN == 0):
    accuracy = 0
else:
    accuracy = (TP + TN)/(TP + TN + FP + FN)

print("Precision : ", precision )
print("Recall : ", recall)
print("F1 : ", F1)
print("Accuracy : ", accuracy)
print("False(+) : ", FP)
print("False(-) : ", FN)

confusion_matrix = [[TN , FP] , [FN , TP]]

mp.imshow(confusion_matrix)
mp.colorbar()   
mp.title("Confusion Matrix")
mp.xlabel("Predicted values")
mp.ylabel("Actual values")
mp.xticks([0,1],["0" , "1"])
mp.yticks([0,1],["0" , "1"])
for i in range(2):
    for j in range(2):
        mp.text(j , i , confusion_matrix[i][j] , ha = "center", va = "center")

mp.show()