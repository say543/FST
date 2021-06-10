import pandas as pd
import math
import argparse
from random import randrange
import random
from tqdm import tqdm

def buketize(file, outputFileDist, outputFile, sampleSize, domain):
	buckets = {}
	bucketLabel = {}
	labels = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
	for i in range(10):
		buckets[i] = []
		bucketLabel[i] = labels[i]

	df = pd.read_csv(file, sep='\t', header=0, encoding = "ISO-8859-1")
	# change bucketizer logic
	#for i, we in enumerate(df[domain]):
	#	if 0<=math.floor(we*10)<10:
	#		buckets[math.floor(we*10)].append((df['Query'][i], we))
	#	if math.floor(we*10)==10:
	#		buckets[9].append((df['Query'][i], we))
	for i, row in tqdm(df.iterrows()):
                row_score = float(row['Score'])
                row_query = row['Query']
                row_domain = row['Domain']

                # skip out of odomain query
                if row_domain != domain:
                        # fro debug	
                        #print("skip OOD query {} against domain:{}".format(row_query, domain))
                        continue

                #print("add {} with score {} ".format(row_query, row_score))
                
                if 0<=math.floor(row_score*10)<10:
                        # fro debug					
                        #print("key {} ".format(math.floor(row_score*10)))
                        buckets[math.floor(row_score*10)].append((row_query, row_score))
                if math.floor(row_score*10)==10:
                        buckets[9].append((row_query, row_score))


			
	f = open(outputFileDist, 'w')
	f.writelines('Bucket\tNumOfQueries\n')
	for key, value in buckets.items():
		f.writelines('{}\t{}\n'.format(bucketLabel[key], len(value)))
	f.close()

	print ('start sampling')
	res = []
	f = open(outputFile,'w', encoding='utf-8')
	headers = ['MessageId', 'GuidelineDomain']
	f.writelines('UserQuery\tBucketGroup\tFrequency\tBucketWeight\tDomainConfidenceScore\tGuidelineDomain\n')
	Freq = {}

	# for debug
	for bucket_id in range(10):
		print("len of bucket{} : {} ".format(bucket_id, len(buckets[bucket_id])))


	while sampleSize:
		# get random bucket
		bucket_id = randrange(10)

		
		(query, score) = random.choice(buckets[bucket_id])
		Freq[(bucket_id, (query, score))] = Freq.get((bucket_id, (query, score)), 0) + 1
		sampleSize -=1
	for key, value_f in Freq.items():
		f.writelines('{}\t{}\t{}\t{}\t{}\t{}\n'.format(key[1][0], key[0], value_f, len(buckets[key[0]]), '%.3f'%(key[1][1]),[domain,"notsure"]))
def arg_parse():
        parser = argparse.ArgumentParser(description='Prepare online Target Sampling file')
        parser.add_argument('--InputFile', type=str, default = 'test.tsv', help = 'Your input raw file')
        parser.add_argument('--SampleSize', type=int, default = 4000, help = 'Your input sample size')
        parser.add_argument('--Domain', type=str, default = 'calendar', help = 'Your input sample size')
        parser.add_argument('--OutputFileDist', type=str, default = 'test.tsv', help = 'Your output stats')
        parser.add_argument('--OutputFile', type=str, default = 'test.tsv', help = 'Your output raw file')
    
        args = parser.parse_args()
        return args

if __name__ == '__main__':
        #args = arg_parse()
		# aether version
		args = arg_parse()
		buketize(args.InputFile, args.OutputFileDist, args.OutputFile, args.SampleSize, args.Domain)

		# local version
        #buketize('E:\\bucket_sampling_aether_test\\BucketSamplingMSB_modified\\MSB_combine_2401.maxent.output.tsv', 'E:\\bucket_sampling_aether_test\\BucketSamplingMSB_modified\\testDist.tsv', 'E:\\bucket_sampling_aether_test\\BucketSamplingMSB_modified\\test.tsv', 200, 'files')
