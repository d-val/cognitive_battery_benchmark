import numpy as np

np.random.seed(1)

tubes = [
	0,  # i only occupies the region under it, it's just vertical
	1,  # it occupies its own region, and 1 addition space to the right
	-1,  # same as above, but to the left
	2,
	-2]

max_trial_bowl_location = 1000
max_trial_tube = 1000
max_new_bowl = 1000

# updates the location of wherever the tube is occupying a space
def update_emptiness(location, tube, bowls):
	if tube < 0:
		for i in range(-tube + 1):
			bowls[location - i] = tube
	elif tube > 0:
		for i in range(tube + 1):
			bowls[location + i] = tube
	else:
		bowls[location] = tube
	return bowls

# checks if a tube can be placed at the current location (current location as well as nearby ones if applicable)
def verify_location(location, tube, bowls):
	direction = np.sign(tube)
	width = np.abs(tube)  # gives the size of the tube, either to left or right

	if direction > 0 and location + width > 7:
		return False
	if direction < 0 and location - width < 0:
		return False
	if width == 0:
		if bowls[location] != 10:
			return False
	for i in range(width + 1):
		if bowls[location + i * direction] != 10:
			return False
	return True

def give_tube_placement(tubes, bowls, n):
	# for n = 8
	if n == 8:
		return [0, 0, 0, 0, 0, 0, 0, 0]

	# for n = 7:
	if n == 7:
		bowls = [0, 0, 0, 0, 0, 0, 0, 0]
		random_tube = np.random.choice(range(2))
		random_location = np.random.choice(range(8))
		if random_tube == 0:
			bowls[random_location] = 10
		else:
			if random_location == 8:
				bowls = [0, 0, 0, 0, 0, 0, 1, 1]
			else:
				bowls[random_location] = 1
				bowls[random_location+1] = 1
		return bowls

	correct_assignment_found = False
	num_full_rest = 0

	while num_full_rest < max_new_bowl and correct_assignment_found == False:
		bowls = [10, 10, 10, 10, 10, 10, 10, 10]
		for n_tubes in range(n):
			counter = 0
			n_trial_bowl_location = 0
			# bowl_location = np.random.choice(range(8))
			while n_trial_bowl_location < max_trial_bowl_location and correct_assignment_found == False:
				# current_tube = np.random.choice(tubes)
				bowl_location = np.random.choice(range(8))
				if bowls[bowl_location] == 10:
					# current_tube = np.random.choice(tubes)
					n_trial_tube = 0
					while n_trial_tube < max_trial_tube and correct_assignment_found == False:
						current_tube = np.random.choice(tubes)
						if verify_location(bowl_location, current_tube, bowls):
							# bowls[bowl_location] = current_tube
							bowls = update_emptiness(bowl_location, current_tube, bowls)
							print("bowls", bowls, "num_full_rest", num_full_rest, "bowl_location", bowl_location)
							counter += 1
							print("counter", counter)
							if counter == n:
								print("tube placement found")
								return bowls
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
	return bowls, counter

"""def record_trials(bowls, tubes, n):
	bowl, counter = give_tube_placement(bowls, tubes, n)
	print("The final placement is:", bowl)
	print("Intended to place", n, "tubes, and", counter, "tubes are placed")
	results.append([bowl, counter])
	print(results)"""

print(give_tube_placement(tubes, [10, 10, 10, 10, 10, 10, 10, 10], 6))

# TOOD: break, continue, ...? statements to escap loops in pythong