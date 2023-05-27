import csv


def reader(filename, firstField, secondField, output):
    data = []
    names = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                names = row
            else:
                data.append(row)
            line_count += 1

    variable1 = names.index(firstField)
    gdp = [float(data[i][variable1]) for i in range(len(data))]

    variable2 = names.index(secondField)
    freedom = [float(data[i][variable2]) for i in range(len(data))]

    variable3 = names.index(output)
    outputs = [float(data[i][variable3]) for i in range(len(data))]

    return gdp, outputs, freedom
