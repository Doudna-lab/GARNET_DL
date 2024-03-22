rfam_hit_file = 'RF00001-4192_stats_evalue_filtered_1.list'
rfam_output_file = 'RF00001-4192_stats_overlaps_removed_2.list'


# dict for checking overlaps
overlap_dict = dict()

# going through each hit and check if it's overlapping or not
with open(rfam_hit_file, 'r') as rf:
	for line in rf:
		info = line.split()
		start = min(int(info[1]), int(info[2]))
		end = max(int(info[1]), int(info[2]))
		evalue = float(info[5])
		contig = info[3]
		#
		# check if entry already exists, if not add it and proceed to next
		try:
			contig_hits = overlap_dict[contig]
		except KeyError:
			overlap_dict[contig] = [[start, end, evalue, line]]
			continue
		#
		# if there's already hits to this contig, check if this one is overlapping with any and has lower e-value
		add_new = True
		add_index = -1
		for i, hit in enumerate(contig_hits):
			c_start = hit[0]
			c_end = hit[1]
			c_evalue = hit[2]
			c_line = hit[3]
			#
			# if overlapping
			if ((end>c_start and end<c_end) or (start>c_start and start<c_end) or (start<c_start and end>c_end) or (start>c_start and end<c_end)) and (c_evalue < evalue):
				add_new = False
				break
			elif ((end>c_start and end<c_end) or (start>c_start and start<c_end) or (start<c_start and end>c_end) or (start>c_start and end<c_end)) and (c_evalue >= evalue):
				add_new = True
				print('%s, %s' % (c_evalue, evalue))
				add_index = i
				break
		#
		if add_new:
			if add_index < 0:
				overlap_dict[contig].append([start, end, evalue, line])
			else:
				overlap_dict[contig][add_index] = [start, end, evalue, line]
				print('replacement of %s with %s' % (c_line, line))


# initializing file
contigs = overlap_dict.keys()

with open(rfam_output_file, 'w') as of:
	for contig in contigs:
		hits = overlap_dict[contig]
		for hit in hits:
			dum = of.write(hit[3])


