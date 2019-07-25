import math
import random
from matplotlib import pyplot as plt
from matplotlib import patches as patch
import numpy as np
from scipy.stats import norm
from scipy import integrate
import scipy.misc as smisc
import colorsys
#TODO: ok but FRACTAL bro
	#all direction splash with shoreline pattern

def add_2d_np_array_side_padding(original, left_width, right_width):

	l = len(original[0])

	left_padding = np.empty((left_width, original[0].size))
	right_padding = np.empty((right_width, original[0].size))


	for i, row in enumerate(original):
		left_padding[:, i] = original[i][0]
		right_padding[:, i] = original[i][l-1]


	print('l ', left_padding.shape)
	print('0 ', original.shape)
	print('r ', right_padding.shape)
	print('b ', np.append(left_padding, original).shape)
	print('all ', np.append(np.append(left_padding, original), right_padding).shape)

	padded_array = np.append(np.append(left_padding, original), right_padding)
	return padded_array

def point_density_value_pdf(x):
	point_density = 0 if x>1 or x<0 else -1*(math.cos(x*1*math.pi)) + 1
	#point_density = 0 if x>1 or x<0 else (-1/3)*(math.cos(3*math.pi*(x+0.175))) + 0.93
	return point_density

def point_density_value_cdf(x):
	integrate.quad
	cumn_prb, error = integrate.quad(lambda e: point_density_value_pdf(e), 0, x)
	return min(cumn_prb, 1)

def hls_to_plt_rgb(h, l, s):
	(r, g, b) = colorsys.hls_to_rgb(h, l, s)
	return [r, g, b]
#adj should be less than 0.5, 0<val<1
def slightly_adjust_unitized_bounce(val, adj, bounce=True, decr_probability=0.5, lower=0, upper=1):
	if random.random()<decr_probability:
		adj *= -1
	val += adj

	#bouce of 0-1 boundary
	if (val<lower or val>upper):
		if bounce:
			val -= 2*adj
		else:
			val = upper if val>upper else lower

	return val

# returns a np.array (size w (width)) of values spanning 3 std (std = w/6), adding to 1
def make_gaussian(std):
	filter = np.fromiter((norm.pdf(i/std) for i in range(-3*std, 3*std+1)), float)
	#filter = [norm.pdf(i/std) for i in range(-3*std, 3*std+1)]
	filter /= np.sum(filter)
	return filter

def block_np_arr_to_px_np_arr(block_arr, block_width, px_width):
	l = len(block_arr)
	px_arr = np.zeros(l * block_width)
	for i in range(l):
		px_arr[block_width*i: block_width*(i+1)] = block_arr[i]
	return px_arr

#row is a numpy array
def apply_blur(row, radius=None):


	l = row.size

	if radius is None:
		radius = max(l//60, 1)
		print('r ', radius)

	filter = make_gaussian(radius)

	print('row ', row.shape, row)

	padded_row = np.append(np.full_like(filter, row[0]), row)
	padded_row = np.append(padded_row, np.full_like(filter, row[l-1]))

	fl = len(filter)
	new_row = np.zeros(len(row))


	#todo: optimize by sliding instead of reslicing
	for x in range(len(row)):

		sum = np.sum(padded_row[x:fl+x] * filter)
		new_row[x] = sum

	return new_row

def np_to_excel(arr, f_name):
	np.savetxt(f_name, arr, delimiter=',')


xn = 50
yn = 300
x_scale = 30
px_ct_x = x_scale*xn # be careful changing, my screen is like 1700 pixels wide ish
px_ct_y = 5*yn
linsp_size = 1001
linsp = np.linspace(0, 1, linsp_size)
linsp = [round(l, 3) for l in linsp]
linsp_intv = linsp[1] - linsp[0]
pdf_pts = list(map(lambda e: point_density_value_pdf(e), linsp))
#plt.plot(linsp, pdf_pts)
#plt.title('pdf and cdf of color value distribution')
#plt.show()
cdf_pts = list(map(lambda e: round(point_density_value_cdf(e), 3), linsp))
inv_norm_dict = dict(zip(cdf_pts, linsp))
#plt.plot(linsp, cdf_pts)
#plt.plot(linsp, [1]*len(linsp))
#plt.show()
#print('intv', linsp_intv)
for l in linsp:
	if l not in inv_norm_dict:
		l_approx = round(l - linsp_intv, 3)
		while l_approx not in inv_norm_dict:
			l_approx -= linsp_intv
			l_approx = round(l_approx, 3)
		inv_norm_dict[l] = inv_norm_dict[l_approx]
print('should be done')
(keys,values) = zip(*inv_norm_dict.items())
#plt.figure()
#plt.title = 'test'
lists = [[k, inv_norm_dict[k]] for k in inv_norm_dict]
lists.sort(key=lambda x: x[0])

#for l in lists:
#	print(l)
#plt.plot(lists)


#plt.show()
print('should be done')
filter = make_gaussian(10)
print(sum(filter))
print(len(filter))
fig2 = plt.figure(figsize=(12,4))
ax = fig2.add_subplot(1, 1, 1)
ax.axis('off')


def sample_color(p):
	r = round(p*1000)
	if r<0 or r>len(lists):
		r = len(lists)/2
		print('btw out of range')
	return lists[r][0]

def rectangles_sample_color():

	n = 12
	ss = []
	for i in range(n):
		s = sample_color(random.random())
		ss.append(s)
		ax.add_patch(
		     patch.Rectangle(
		        (i*(1/n), 0.05),
		        0.7/n,
		        0.9,
		        facecolor=hls_to_plt_rgb(0.95, 0.3+(0.4*s), 0.15+(0.85*s))
		        #facecolor=hls_to_plt_rgb(0.53, 0.3+(0.65*s), 0.15+(0.85*s))
		        #facecolor=hls_to_plt_rgb( 0.5+(0.1*s), 0.3+(0.4*s), 0.15+(0.85*s))
		        #fill=False      # remove background
		     ) ) 
	#fig2.show()
	plt.show()
	#num_bins = 300
	#n, bins, patches = plt.hist(ss, num_bins, facecolor='blue', alpha=0.5)
	#plt.show()

def gradientey_lines():
	n = 200
	s = sample_color(random.random())
	for i in range(n):
		adj = 0.1 * sample_color(random.random())
		s = slightly_adjust_unitized_bounce(s, adj)
		ax.add_patch(
		     patch.Rectangle(
		        (0, i*(1/n)),
		        1,
		        1/n,
		        facecolor=hls_to_plt_rgb(0.53, 0.2+(0.75*s), 0.1+(0.75*s))
		        #facecolor=hls_to_plt_rgb( 0.5+(0.1*s), 0.3+(0.4*s), 0.15+(0.85*s))
		        #fill=False      # remove background
		     ) ) 
	#fig2.show()
	plt.show()
	file_rand_suffix = random.randint(1, 100000000000)
	f_name='C:/Users/Jessie Steckling/Desktop/colorgen/teals'+str(file_rand_suffix)+'.png'
	fig2.savefig(f_name)

def splash_right():
	yn = 150
	xn = 20
	s = sample_color(random.random())


	for i in range(yn):
		x_offset = random.random()/xn
		adjs = 0.10 * sample_color(random.random())
		s = slightly_adjust_unitized_bounce(s, adjs)
		t=s
		for j in range(xn+1):
			adjt = 0.10 * sample_color(random.random())
			t = slightly_adjust_unitized_bounce(t, adjt)
			ax.add_patch(
			     patch.Rectangle(
			        (j*(1/xn) - x_offset, i*(1/yn)),
			        1/xn,
			        1/yn,
			        #facecolor=hls_to_plt_rgb(0.53, 0.25+(0.7*t), 0.15+(0.75*t))
			        facecolor=hls_to_plt_rgb(0.53, 0.2+(0.75*t), 0.1+(0.75*t))
			        #facecolor=hls_to_plt_rgb( 0.5+(0.1*s), 0.3+(0.4*s), 0.15+(0.85*s))
			        #fill=False      # remove background
		     	) ) 
	#fig2.show()
	plt.show()
	file_rand_suffix = random.randint(1, 100000000000)
	f_name='C:/Users/Jessie Steckling/Desktop/colorgen/teals'+str(file_rand_suffix)+'.png'
	fig2.savefig(f_name)

def splash_inward_splotch():
	yn = 200
	xn = 20
	dark_lightness = 0.2
	light_lightness = 0.9
	hue = 0.53
	saturation = 0.4

	s = sample_color(random.random())

	#fill in background
	ax.add_patch(
     patch.Rectangle(
        (0, 0),
        1,
        1,
        #facecolor=hls_to_plt_rgb(0.53, 0.25+(0.7*t), 0.15+(0.75*t))

        #good possible color spacea
        #facecolor=hls_to_plt_rgb(0.53, 0.6, 0.4)
        #facecolor=hls_to_plt_rgb(0.53, 0.9, 0.4)

        facecolor=hls_to_plt_rgb(hue, light_lightness, saturation)
 	))
 	#generate offsets
	x_offsets = [1/xn*random.random() for i in range(yn)]

 	#x_offsets = [0.1]*200
 	#generate splotch locations
	y_origins = []
	y = math.floor(yn*0.2*sample_color(random.random()))
	while y<yn:
		y_origins.append((y, 0 if random.random()<0.35 else 1))
		y += math.floor(yn*0.4*sample_color(random.random()))
	print('splotches: ', y_origins)
	#splotch at splotch locations
	for y_o, is_left in y_origins:
		for d in [-1, 1]:
			y_cur = y_o
			print('y_cur===', y_cur, ' left :', is_left,' d: ', d, ' offs: ', x_offsets[y_cur])

			#move vertivally
			l = dark_lightness
			#todo: mayve have col @x=0 oscilate, then decrease

			while 0<y_cur<yn and l < light_lightness:

				j = -1 if is_left else xn+1
				l_x = l

				# move horzontally
				while -2<j<(xn+2) and l_x<light_lightness:

					#todo: add x_offsets
					ax.add_patch(
				     patch.Rectangle(
				        (x_offsets[y_cur]+ j*(1/xn) , y_cur/yn),
				        1/xn,
				        1/yn,
				        #facecolor=hls_to_plt_rgb(0.53, 0.25+(0.7*t), 0.15+(0.75*t))
				        facecolor=hls_to_plt_rgb(hue, l_x, saturation)
				        #facecolor=hls_to_plt_rgb( 0.5+(0.1*s), 0.3+(0.4*s), 0.15+(0.85*s))
				        #fill=False      # remove background
			     	) ) 			     	
					adjl_x = 0.3 * sample_color(random.random())
					l_x = slightly_adjust_unitized_bounce(l_x, adjl_x, False, 0.05)

					if is_left:
						j += 1
					else:
						j -= 1



				y_cur += 1
				adjl = 0.1 * sample_color(random.random())
				l = slightly_adjust_unitized_bounce(l, adjl)


	
	#fig2.show()
	plt.show()
	file_rand_suffix = random.randint(1, 100000000000)
	f_name='C:/Users/Jessie Steckling/Desktop/colorgen/teals'+str(file_rand_suffix)+'.png'
	fig2.savefig(f_name)

	#	point_density = 0 if x>1 or x<0 else -1*(math.cos(x*2*math.pi)) + 1

def splash_inward():

	dark_lightness = 0.2
	light_lightness = 0.9
	hue = 0.53
	saturation = 0.4

	s = sample_color(random.random())

	#fill in background
	ax.add_patch(
     patch.Rectangle(
        (0, 0),
        1,
        1,
        #facecolor=hls_to_plt_rgb(0.53, 0.25+(0.7*t), 0.15+(0.75*t))

        #good possible color spacea
        #facecolor=hls_to_plt_rgb(0.53, 0.6, 0.4)
        #facecolor=hls_to_plt_rgb(0.53, 0.9, 0.4)

        facecolor=hls_to_plt_rgb(hue, light_lightness, saturation)
 	))
 	#generate offsets
	x_offsets = [1/xn*random.random() for i in range(yn)]

	#move vertivally
	l = random.random()*(light_lightness - dark_lightness) + dark_lightness
	#todo: mayve have col @x=0 oscilate, then decrease

	# move along right side then left
	for is_left in [False, True]: #(False, True):

		y_cur = 0
		rows = np.full((yn, px_ct_x), light_lightness)

		#move vertically
		while y_cur<yn:

			j = xn+1 if is_left else -1
			l_x = l
			row = rows[y_cur]
			rect_dx_px = math.floor(1/xn*px_ct_x)

			# move horzontally
			while -2<j<(xn+2) and l_x<light_lightness:
				rect_x0 = x_offsets[y_cur]+ (j*(1/xn))
				rect_x0_px = math.floor((rect_x0)*px_ct_x)
				if 0<rect_x0<1:
					row[rect_x0_px : rect_x0_px + rect_dx_px] = l_x

				#todo: add x_offsets
				ax.add_patch(
			     patch.Rectangle(
			        (rect_x0 , y_cur/yn),
			        1/xn,
			        1/yn,
			        #facecolor=hls_to_plt_rgb(0.53, 0.25+(0.7*t), 0.15+(0.75*t))
			        facecolor=hls_to_plt_rgb(hue, l_x, saturation)
			        #facecolor=hls_to_plt_rgb( 0.5+(0.1*s), 0.3+(0.4*s), 0.15+(0.85*s))
			        #fill=False      # remove background
		     	) ) 			     	
				adjl_x = 0.15 * sample_color(random.random())
				l_x = slightly_adjust_unitized_bounce(l_x, adjl_x, False, 0.2)

				if is_left:
					j -= 1
				else:
					j += 1

			y_cur += 1
			adjl = 0.1 * sample_color(random.random())
			l = slightly_adjust_unitized_bounce(l, adjl, upper=light_lightness, lower=dark_lightness)

	plt.show()

	file_rand_suffix = random.randint(1, 100000000000)
	f_name='C:/Users/Jessie Steckling/Desktop/colorgen/teals'+str(file_rand_suffix)+'.png'
	#fig2.savefig(f_name)

	cut_off = math.floor(1/xn*px_ct_x)
	rows = rows[:, cut_off:-cut_off]

	gaus_blurred_rows = np.empty_like(rows)

	for i, r in enumerate(rows):
		gaus_blurred_rows[i] = apply_blur(r, 5)
		print

	(height, width) = gaus_blurred_rows.shape
	h_stretch = 3
	pixel_grid_l = np.empty((height*h_stretch, width))
	for i in range(height):
		pixel_grid_l[h_stretch*i:h_stretch*(i+1)] = gaus_blurred_rows[i]


	#gaus_blurred_rows_2d = np.empty_like(pixel_grid_l)
	#for j in range(pixel_grid_l[0].size):
	#	col = pixel_grid_l[:, j]
	#	new  = apply_blur(col, 5)
	#	gaus_blurred_rows_2d[:, j] = new


	grid_3_channel = np.empty((height, width, 3))
	for i, r in enumerate(gaus_blurred_rows):
		for j, c in enumerate(r):
			grid_3_channel[i][j] = np.array(hls_to_plt_rgb(hue, c, saturation))

	print(grid_3_channel.shape)
	img = smisc.toimage(grid_3_channel)
	img.show()
	smisc.imsave(f_name, img)

	

	show = False
	if show==True:
		#fig2.show()
		plt.show()
		print('shown')

		fig3 = plt.figure(figsize=(12,5))
		ax3 = fig3.add_subplot(1, 1, 1)
		px_width = 1/px_ct_x
		for i, row in enumerate(rows):
			print('on ', i)
			for j, l in enumerate(row):
				#todo: add x_offsets
				ax3.add_patch(
			     patch.Rectangle(
			        (j*px_width , i/yn),
			        px_width,
			        1/yn,
			        #facecolor=hls_to_plt_rgb(0.53, 0.25+(0.7*t), 0.15+(0.75*t))
			        facecolor=hls_to_plt_rgb(.9, l, saturation)
			        #facecolor=hls_to_plt_rgb( 0.5+(0.1*s), 0.3+(0.4*s), 0.15+(0.85*s))
			        #fill=False      # remove background
		     	))
		plt.show()
		print('shown')

def paintbrush_with_diag():
	dark_lightness = 0.2
	light_lightness = 0.9
	hue = 0.53
	saturation = 0.4

	s = sample_color(random.random())

	#fill in background
	ax.add_patch(
     patch.Rectangle(
        (0, 0),
        1,
        1,
        #facecolor=hls_to_plt_rgb(0.53, 0.25+(0.7*t), 0.15+(0.75*t))

        #good possible color spacea
        #facecolor=hls_to_plt_rgb(0.53, 0.6, 0.4)
        #facecolor=hls_to_plt_rgb(0.53, 0.9, 0.4)

        facecolor=hls_to_plt_rgb(hue, light_lightness, saturation)
 	))
 	#generate offsets
	x_offsets = [1/xn*random.random() for i in range(yn)]

	#move vertivally
	l = random.random()*(light_lightness - dark_lightness) + dark_lightness
	#todo: mayve have col @x=0 oscilate, then decrease

	# move along right side then left
	for is_left in [False]: #(False, True):

		y_cur = 0
		rows = np.full((yn, px_ct_x), light_lightness)

		#move vertically
		while y_cur<yn:

			j = xn+1 if is_left else -1
			l_x = l
			row = rows[y_cur]
			rect_dx_px = math.floor(1/xn*px_ct_x)

			# move horzontally
			while -2<j<(xn+2) and l_x<light_lightness:
				rect_x0 = x_offsets[y_cur]+ (j*(1/xn))
				rect_x0_px = math.floor((rect_x0)*px_ct_x)
				if 0<rect_x0<1:
					row[rect_x0_px : rect_x0_px + rect_dx_px] = l_x

				#todo: add x_offsets
				ax.add_patch(
			     patch.Rectangle(
			        (rect_x0 , y_cur/yn),
			        1/xn,
			        1/yn,
			        #facecolor=hls_to_plt_rgb(0.53, 0.25+(0.7*t), 0.15+(0.75*t))
			        facecolor=hls_to_plt_rgb(hue, l_x, saturation)
			        #facecolor=hls_to_plt_rgb( 0.5+(0.1*s), 0.3+(0.4*s), 0.15+(0.85*s))
			        #fill=False      # remove background
		     	) ) 			     	
				adjl_x = 0.15 * sample_color(random.random())


				if 0<rect_x0<1:
					row[rect_x0_px : rect_x0_px + rect_dx_px] = l_x

					if rect_x0_px < x_scale:
						l_x = slightly_adjust_unitized_bounce(l_x, adjl_x, False, 0.2)
					else:
						l_x_a = slightly_adjust_unitized_bounce(l_x, adjl_x, False, 0.2)
						l_x_b = row[rect_x0_px-1]
						p = 0.5+(0.35*random.random())
						l_x = (p*l_x_a) + ((1-p)*l_x_b)

				if is_left:
					j -= 1
				else:
					j += 1

			y_cur += 1
			adjl = 0.1 * sample_color(random.random())
			l = slightly_adjust_unitized_bounce(l, adjl, upper=light_lightness, lower=dark_lightness)

	plt.show()

	file_rand_suffix = random.randint(1, 100000000000)
	f_name='C:/Users/Jessie Steckling/Desktop/colorgen/teals'+str(file_rand_suffix)+'.png'
	#fig2.savefig(f_name)

	cut_off = math.floor(1/xn*px_ct_x)
	rows = rows[:, cut_off:-cut_off]

	gaus_blurred_rows = np.empty_like(rows)

	for i, r in enumerate(rows):
		gaus_blurred_rows[i] = apply_blur(r, 5)
		print

	(height, width) = gaus_blurred_rows.shape
	h_stretch = 3
	pixel_grid_l = np.empty((height*h_stretch, width))
	for i in range(height):
		pixel_grid_l[h_stretch*i:h_stretch*(i+1)] = gaus_blurred_rows[i]


	#gaus_blurred_rows_2d = np.empty_like(pixel_grid_l)
	#for j in range(pixel_grid_l[0].size):
	#	col = pixel_grid_l[:, j]
	#	new  = apply_blur(col, 5)
	#	gaus_blurred_rows_2d[:, j] = new


	grid_3_channel = np.empty((height, width, 3))
	for i, r in enumerate(gaus_blurred_rows):
		for j, c in enumerate(r):
			grid_3_channel[i][j] = np.array(hls_to_plt_rgb(hue, c, saturation))

	print(grid_3_channel.shape)
	img = smisc.toimage(grid_3_channel)
	img.show()
	smisc.imsave(f_name, img)


	show = False
	if show==True:
		#fig2.show()
		plt.show()
		print('shown')

		fig3 = plt.figure(figsize=(12,5))
		ax3 = fig3.add_subplot(1, 1, 1)
		px_width = 1/px_ct_x
		for i, row in enumerate(rows):
			print('on ', i)
			for j, l in enumerate(row):
				#todo: add x_offsets
				ax3.add_patch(
			     patch.Rectangle(
			        (j*px_width , i/yn),
			        px_width,
			        1/yn,
			        #facecolor=hls_to_plt_rgb(0.53, 0.25+(0.7*t), 0.15+(0.75*t))
			        facecolor=hls_to_plt_rgb(.9, l, saturation)
			        #facecolor=hls_to_plt_rgb( 0.5+(0.1*s), 0.3+(0.4*s), 0.15+(0.85*s))
			        #fill=False      # remove background
		     	))
		plt.show()
		print('shown')

def rope_fray():

	dark_lightness = 0.2
	light_lightness = 0.9
	hue = 0.53
	saturation = 0.4

	s = sample_color(random.random())

	#fill in background
	ax.add_patch(
     patch.Rectangle(
        (0, 0),
        1,
        1,
        #facecolor=hls_to_plt_rgb(0.53, 0.25+(0.7*t), 0.15+(0.75*t))

        #good possible color spacea
        #facecolor=hls_to_plt_rgb(0.53, 0.6, 0.4)
        #facecolor=hls_to_plt_rgb(0.53, 0.9, 0.4)

        facecolor=hls_to_plt_rgb(hue, light_lightness, saturation)
 	))
 	#generate offsets
	x_offsets = [1/xn*random.random() for i in range(yn)]

	#move vertivally
	l = random.random()*(light_lightness - dark_lightness) + dark_lightness
	#todo: mayve have col @x=0 oscilate, then decrease

	# move along right side then left
	for is_left in [False]: #(False, True):

		y_cur = 0
		rows = np.full((yn, px_ct_x), light_lightness)

		#move vertically
		while y_cur<yn:

			j = xn+1 if is_left else -1
			l_x = l
			row = rows[y_cur]
			rect_dx_px = math.floor(1/xn*px_ct_x)

			# move horzontally
			while -2<j<(xn+2) and l_x<light_lightness:
				rect_x0 = x_offsets[y_cur]+ (j*(1/xn))
				rect_x0_px = math.floor((rect_x0)*px_ct_x)
				if 0<rect_x0<1:
					row[rect_x0_px : rect_x0_px + rect_dx_px] = l_x

				#todo: add x_offsets
				ax.add_patch(
			     patch.Rectangle(
			        (rect_x0 , y_cur/yn),
			        1/xn,
			        1/yn,
			        #facecolor=hls_to_plt_rgb(0.53, 0.25+(0.7*t), 0.15+(0.75*t))
			        facecolor=hls_to_plt_rgb(hue, l_x, saturation)
			        #facecolor=hls_to_plt_rgb( 0.5+(0.1*s), 0.3+(0.4*s), 0.15+(0.85*s))
			        #fill=False      # remove background
		     	) ) 			     	
				adjl_x = 0.15 * sample_color(random.random())
				l_x = slightly_adjust_unitized_bounce(l_x, adjl_x, False, 0.2)

				if is_left:
					j -= 1
				else:
					j += 1

			y_cur += 1
			adjl = 0.1 * sample_color(random.random())
			l = slightly_adjust_unitized_bounce(l, adjl, upper=light_lightness, lower=dark_lightness)

	plt.show()

	file_rand_suffix = random.randint(1, 100000000000)
	f_name='C:/Users/Jessie Steckling/Desktop/colorgen/teals'+str(file_rand_suffix)+'.png'
	#fig2.savefig(f_name)

	cut_off = math.floor(1/xn*px_ct_x)
	rows = rows[:, cut_off:-cut_off]

	print('rows: ', rows.shape, rows)
	rows = add_2d_np_array_side_padding(rows, 200, 100)

	gaus_blurred_rows = np.empty_like(rows)

	print('rows: ', rows.shape, rows)

	for i, r in enumerate(rows):
		gaus_blurred_rows[i] = apply_blur(r, 5)
		print

	(height, width) = gaus_blurred_rows.shape
	h_stretch = 3
	pixel_grid_l = np.empty((height*h_stretch, width))
	for i in range(height):
		pixel_grid_l[h_stretch*i:h_stretch*(i+1)] = gaus_blurred_rows[i]


	#gaus_blurred_rows_2d = np.empty_like(pixel_grid_l)
	#for j in range(pixel_grid_l[0].size):
	#	col = pixel_grid_l[:, j]
	#	new  = apply_blur(col, 5)
	#	gaus_blurred_rows_2d[:, j] = new


	grid_3_channel = np.empty((height, width, 3))
	for i, r in enumerate(gaus_blurred_rows):
		for j, c in enumerate(r):
			grid_3_channel[i][j] = np.array(hls_to_plt_rgb(hue, c, saturation))

	print(grid_3_channel.shape)
	img = smisc.toimage(grid_3_channel)
	img.show()
	smisc.imsave(f_name, img)

	

	show = False
	if show==True:
		#fig2.show()
		plt.show()
		print('shown')

		fig3 = plt.figure(figsize=(12,5))
		ax3 = fig3.add_subplot(1, 1, 1)
		px_width = 1/px_ct_x
		for i, row in enumerate(rows):
			print('on ', i)
			for j, l in enumerate(row):
				#todo: add x_offsets
				ax3.add_patch(
			     patch.Rectangle(
			        (j*px_width , i/yn),
			        px_width,
			        1/yn,
			        #facecolor=hls_to_plt_rgb(0.53, 0.25+(0.7*t), 0.15+(0.75*t))
			        facecolor=hls_to_plt_rgb(.9, l, saturation)
			        #facecolor=hls_to_plt_rgb( 0.5+(0.1*s), 0.3+(0.4*s), 0.15+(0.85*s))
			        #fill=False      # remove background
		     	))
		plt.show()
		print('shown')



rope_fray()