# file_val = open('zurich_val.txt', 'r')
# file_val_ref = open('zurich_val_ref.txt', 'w+')
#
# Lines = file_val.readlines()
# for line in Lines:
#     line=line.replace('night','day')
#     line=line.replace('GOPR0356','GOPR0356_ref',1)
#     line=line.replace('_rgb','_ref_rgb')
#     file_val_ref.write(line)

file_test = open('zurich_test.txt', 'r')
file_test_ref = open('zurich_test_ref.txt', 'w+')

Lines = file_test.readlines()
for line in Lines:
    line=line.replace('night','day')
    line=line.replace('GOPR0364','GOPR0364_ref',1)
    line=line.replace('GP010364','GP010364_ref',1)

    line=line.replace('_rgb','_ref_rgb')
    file_test_ref.write(line)