from datasets import load_dataset
import csv

dataset = load_dataset('glue', 'mrpc', split='train')
my_iter = iter(dataset)

with open('mrpc_data.csv', 'w+') as output:
    writer = csv.writer(output)
    writer.writerow(['section_one', 'section_two', 'label'])
    count_ones = 0
    count_zeroes = 0
    for x in my_iter:

        label = x['label']
        if label == 1:
            count_ones += 1
        if label == 0:
            count_zeroes += 1

print(count_ones, "count_ones")
print(count_zeroes, "count_zeroes")








