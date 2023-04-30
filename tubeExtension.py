import numpy as np

np.random.seed(5)

bowls = [10, 10, 10, 10, 10, 10, 10, 10]

tubes = [
	0,  # i only occupies the region under it, it's just vertical
	1,  # it occupies its own region, and 1 addition space to the right
	-1,  # same as above, but to the left
	2,
	-2
]

occupy_flag = 1000
results = []

max_trial_bowl_location = 50
max_trial_tube = 1000
max_new_bowl = 10

"""def update_emptiness(location, tube, bowls):
	# it updates the location of wherever the tube is occupying a space
	if tube < 0:
		# bowls[location+tube: tube] = occupy_flag
		for i in range(-tube+1):
			bowls[location-i] = occupy_flag
	elif tube >0:
		for i in range(tube+1):
			bowls[location+i] = occupy_flag
		# bowls[location: location+tube] = occupy_flag
	else:
		bowls[location] = occupy_flag 
	return bowls"""


def update_emptiness(location, tube, bowls):
	# it updates the location of wherever the tube is occupying a space
	if tube < 0:
		# bowls[location+tube: tube] = occupy_flag
		for i in range(-tube + 1):
			bowls[location - i] = tube
	elif tube > 0:
		for i in range(tube + 1):
			bowls[location + i] = tube
	# bowls[location: location+tube] = occupy_flag
	else:
		bowls[location] = tube

	# counter_tubes_placed = counter_tubes_placed + 1
	return bowls #, counter_tubes_placed


"""def verify_location(location, tube):
	# verify if the locations the tube WOULD be over are all empty
	direction = np.sign(tube) # +1 or -1
	# tube is the current tube 
	width = np.abs(tube) #3 
	is_possible = False 

	for position in range(width): # go forward or backwards
		if location-position < 0 or location-position > len(bowls):
			is_possible = False
		else:
			is_possible = True
	return is_possible"""

# this is only checking the current index. Need to check +-1, +-2
def verify_location(location, tube, bowls):
	# location is the current location
	# tube is the chosen tube: -2, -1, 0, 1, 2
	direction = np.sign(tube)
	width = np.abs(tube)  # gives the size of the tube, either to left or right
	is_possible = False
	"""""# positive tube
	if direction > 0 and location + width > 7:
		is_possible = False
	elif direction < 0 and location - width < 0:
		is_possible = False
	else:
		is_possible = True
	return is_possible"""
	if direction > 0 and location + width > 7:
		return False
	if direction < 0 and location - width < 0:
		return False
	if width == 0:
		if bowls[location] != 10:
			return False
	for i in range(width):
		if bowls[location+i*direction] != 10:
			return False

	"""if width == 1:
		if bowls[location+tube] == occupy_flag:
			return False
	if width == 2:
		if bowls[location+tube] == occupy_flag:
			return False"""
	return True

def give_tube_placement(tubes, bowls, n):
	correct_assignment_found = False
	num_full_rest = 0
	while num_full_rest < max_new_bowl and correct_assignment_found == False:
		bowls = [10, 10, 10, 10, 10, 10, 10, 10]
		counter = 0
		for n_tubes in range(n):
			n_trial_bowl_location = 0
			bowl_location = np.random.choice(range(8))
			while n_trial_bowl_location < max_trial_bowl_location and correct_assignment_found == True:
				# current_tube = np.random.choice(tubes)
				if bowls[bowl_location] != occupy_flag:
					current_tube = np.random.choice(tubes)

					n_trial_tube = 0
					while n_trial_tube < max_trial_tube and correct_assignment_found == True:
						# print("current tube", current_tube, "start location", bowl_location)
						if verify_location(bowl_location, current_tube, bowls):
							# bowls[bowl_location] = current_tube
							bowls = update_emptiness(bowl_location, current_tube, bowls)
							counter += 1
							"""if current_tube < 0:
								bowls[bowl_location + current_tube: bowl_location] = occupy_flag
							elif current_tube > 0:
								bowls[bowl_location: bowl_location + current_tube] = occupy_flag
							else:
								bowls[bowl_location] = occupy_flag"""
						else:
							current_tube = np.random.choice(tubes)
						n_trial_tube = n_trial_tube + 1
				else:
					bowl_location = np.random.choice(range(8))
				n_trial_bowl_location = n_trial_bowl_location + 1
		num_full_rest += 1
		# print(n_trial_bowl_location, n_trial_tube)
		if correct_assignment_found == counter:
			correct_assignment_found = True
	# assert n == counter, 'could not find satisfactory placement'
	return bowls, counter

def record_trials(bowls, tubes, n):
	bowl, counter = give_tube_placement(bowls, tubes, n)
	print("The final placement is:", bowl)
	print("Intended to place", n, "tubes, and", counter, "tubes are placed")
	results.append([bowl, counter])
	print(results)

record_trials(bowls, tubes, 2)

# TOOD: break, continue, ...? statements to escap loops in pythong