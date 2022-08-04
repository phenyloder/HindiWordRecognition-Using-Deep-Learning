import numpy as np

def load_hindi_data(datasetPath):

    data=[]
    labels=[]

    for row in open(datasetPath):

        row = row.split(",")
        label = row[0]
        image = np.array([int(x) for x in row[1:]], dtype="uint8")

        image = image.reshape((32,32))

        data.append(image)
        labels.append(label)

    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    return (data, labels)


