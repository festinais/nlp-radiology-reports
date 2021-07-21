from datasets import load_dataset
import csv

dataset = load_dataset('glue', 'mrpc', split='train')
my_iter = iter(dataset)

with open('mrpc_data.csv', 'w+') as output:
    writer = csv.writer(output)
    writer.writerow(['section_one', 'section_two', 'label'])
    count = 0
    for x in my_iter:
        count += 1
        label = x['label']
        if label == 1:
            print(x['label'])
            writer.writerow([x['sentence1'], x['sentence2'], x['label']])







