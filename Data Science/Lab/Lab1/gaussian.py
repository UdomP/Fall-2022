import numpy as np

def extract_data_salmon(filename): 
# write your code to return mean and std of salmon 
    infile = open(filename, 'r')
    fish = []
    n = 0
    for w in infile.read().split():
        try:
            n = float(w)
        except ValueError:
            continue
        if (n >= 0) and (n <= 100):
            fish.append(n)
    infile.close()
    return np.mean(fish), np.std(fish)


def extract_data_bass(filename): 
# write your code to return mean and std of bass 
    # write your code to return mean and std of salmon 
    infile = open(filename, 'r')
    fish = []
    n = 0
    for w in infile.read().split():
        ww = w.split(",")
        for www in ww:
            try:
                n = float(www)
            except ValueError:
                continue
            if (n >= 0) and (n <= 100):
                fish.append(n)
    infile.close()
    return np.mean(fish), np.std(fish)

def extract_data_trout(filename): 
# write your code to return mean and std of trout 
    # write your code to return mean and std of salmon 
    infile = open(filename, 'r')
    fish = []
    n = 0
    for w in infile.read().split():
        try:
            n = float(w)
        except ValueError:
            continue
        if (n >= 0) and (n <= 100):
            fish.append(n)
    infile.close()
    return np.mean(fish), np.std(fish)

mean1, std1 = extract_data_salmon('Atlantic_salmon.txt') 
mean2, std2 = extract_data_bass('Largemouth_bass.txt') 
mean3, std3 = extract_data_trout('Rainbow_trout.txt') 
print(mean1, std1) 
print(mean2, std2) 
print(mean3, std3)

outfile = open('result.txt', 'w')
outfile.write('Mean and standard deviation of Gaussian of Atlantic salmon are ' + str(mean1) + ' and ' + str(std1) + '.\n')
outfile.write('Mean and standard deviation of Gaussian of Largemouth bass are ' + str(mean2) + ' and ' + str(std2) + '.\n')
outfile.write('Mean and standard deviation of Gaussian of Rainbow trout are ' + str(mean3) + ' and ' + str(std3) + '.\n')
outfile.close()