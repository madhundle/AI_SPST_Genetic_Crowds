# -*- coding: utf-8 -*-
"""
Project 6
CECS 545 Artificial Intelligence
Author: Madeline Hundley

This program uses a genetic algorithm with a wisdom of the crowds approach in
order to find the solution to a Shortest Total-Path-Length Spanning Tree (SPST)
problem. For further information see the accompanying report.

To change the level of logging statements, change the line at the beginning of
the main program. DEBUG is extremely verbose and shows detailed inner workings
of the code. INFO shows the algorithm's steps.
"""

################################## IMPORTS ######################################

import math # required to calculate distances
import pygame # generates graphics
import random # used with probabilities
import logging # generates logging statements
import time # to test algorithm performance
import matplotlib.pyplot as plt # to graph stats
import numpy.random as nprandom # probabilities
from operator import attrgetter # for sorting
from collections import Counter # for counting

######################### ADMIN FUNCTION DEFINITIONS ############################

def read_in_coords(coord_filename=""):
    """
    Function to read in coordinate data from Concorde text files
    Input: none
    Output: (dict) key= vertex num (int):
                   value= x (float), y (float) coordinates (tuple)
    """
    coords = {}
    coord_section = False # flag for identifying NODE_COORD_SECTION
    if coord_filename == "":
        coord_filename = input("Input the filename with"
                               "the vertex coordinate data: ")
    coord_file = open(coord_filename)
    for line in coord_file:
        if coord_section == True: # if in coord section
            v = line.split(' ') # record vertex's coords
            v[2] = v[2].strip('\n') # clean the input
            coords[int(v[0])] = (float(v[1]),float(v[2]))
        if line == "NODE_COORD_SECTION\n": coord_section = True
    coord_file.close()
    return coords

def calc_dists():
    """
    Function to calculate the Euclidean distances between all vertices
    Input: none
    Output: (dict) key= 1st vertex num, 2nd vertex num (tuple):
                   value= edge distance (float)
    """
    dists = {} # empty dictionary
    # range(1,num_vertices+1) produces the list of vertex nums
    for v1 in range(1,NUM_VERTICES+1): # every "from" vertex
        for v2 in range(v1+1,NUM_VERTICES+1): # "to" vertex
            # extract x, y of each vertex
            x1,y1 = COORDS[v1]
            x2,y2 = COORDS[v2]
            # calculate Euclidean distance
            dist = math.sqrt(math.pow((y2-y1),2) + math.pow((x2-x1),2))
            # dictionary is twice as long but ensures simple lookups
            dists[(v1,v2)] = dist
            dists[(v2,v1)] = dist
    return dists

################### ALGORITHM CLASS AND FUNCTION DEFINITIONS ####################

class SpanningTree(object):
    """
    Class for a Spanning Tree in an edge-weighted, undirected graph

    Instance attributes:
    sptree (set) - edges that exist in the spanning tree (tuples)
                   Note: class must be passed unique tuples, but internally it
                         records both directions of vertex-pairs; uses more
                         memory but worth it for simple and efficient lookups to
                         the set

    paths (dict) - key= pair of vertices (u,v) (tuple):
                   value= chain of edges from u to v (list)

    length (float) - total path length

    Class attributes:
    pm - (float) the probability, Pm, between 0 and 1, of a mutation occuring in
                 the "genes" of a sptree
    pc = (float) the probability, Pc, between 0 and 1, of crossover occuring when
                 forming children from the "genes" of parent sptrees
    """

    pm = 0.10 # the probability of mutation
    pc = 0.95 # the crossover probability


    def __init__(self, sptree):
        """
        Initialize a new sptree with its list of vertices and total length
        """
        logging.debug("Initializing Spanning Tree...")
        # Store sptree with all edges both forwards and backwards
        reverseds = {(e[1],e[0]) for e in sptree}
        self.sptree = sptree | reverseds
        self.paths = self.calc_paths() # Determine all paths in the sptree
        self.length = self.calc_length() # Calculate total-path-length
        #logging.debug("SPTree {}".format(self))
        #logging.debug("Paths {}".format(self.paths))
        return

    def __repr__(self):
        """
        Represent a SpanningTree object
        """
        return str(self)

    def __str__(self):
        """
        Print the sptree and its total length
        """
        sptree_str = "SpanningTree "
        sptree_str += str({e for e in self.sptree if e[0]<e[1]})
        sptree_str += " Length {:.5f}".format(self.length)
        return sptree_str

    def __eq__(self, other):
        """
        Returns True if the sptrees contains all the same edges
        """
        return self.sptree == other.sptree

    def calc_paths(self):
        """
        For every pair of vertices, determines the path between
        """
        def recursive_expansion(path):
            #logging.debug("recursive_expansion called on {}".format(path))
            if tuple(path[-2:]) in self.sptree: # recursion base case
                #logging.debug("found path {}".format(path))
                return path
            else:
                frontier = {b for (a,b) in self.sptree if a==path[-2]}
                frontier -= set(path[:-1]) # don't expand "backwards"
                #logging.debug("frontier {}".format(frontier))
                while frontier: # if current path can be expanded, do so
                    path.insert(-1,frontier.pop())
                    attempt = recursive_expansion(path)
                    if attempt == None:
                        path.pop(-2) # hit "dead end", back up and continue
                        #logging.debug("made it to here, path {}".format(path))
                        continue # try next vertex in frontier
                    #logging.debug("returning {}".format(attempt))
                    return attempt
                else:
                    #logging.debug("dead end, popping {}".format(path[-2]))
                    return None # "dead end"
            raise Exception("Recursion Failed")

        paths = {}.fromkeys([(a,b) for a in VERTICES
                             for b in range(a+1,NUM_VERTICES+1)])
        for u,v in paths: # for every pair of edges
            # path[0] is always u, path[-1] is always v
            path = [u,v] # initialize path
            path = recursive_expansion(path)
            paths[(u,v)] = path # save
            #paths[(v,u)] = path # save

        return paths

    def calc_length(self):
        """
        Calculates total-path-length of the spanning tree
        """
        tpl = 0
        for path in self.paths.values():
            for i in range(len(path)-1):
                tpl += DISTS[(path[i],path[i+1])]
        return tpl

    def mutate(self):
        if random.random() <= SpanningTree.pm: # check if mutation occurs
            # get all paths with two edges i.e. three vertices
            options = {p for p in self.paths if len(self.paths[p])==3}
            pair = random.sample(options,1)[0] # choose 1 to change
            path = self.paths[pair] # get the path
            # remove and add as appropriate
            self.sptree.remove((path[1],path[2]))
            self.sptree.remove((path[2],path[1]))
            self.sptree.add((path[0],path[2]))
            self.sptree.add((path[2],path[0]))
            # update sptree's paths and length
            self.paths = self.calc_paths()
            self.length = self.calc_length()
            logging.info("Mutation removed {} and added " \
                         "{}".format((path[1],path[2]),(path[0],path[2])))
        return

def get_rand_sptree():
    """
    Function to randomly generate a new sptree
    Input: none
    Output: A sptree (SpanningTree)
    """
    sptree = set() # placeholder set for the new sptree
    a = random.choice(list(VERTICES))
    b = random.choice(list(VERTICES-{a}))
    sptree.add((a,b))
    #sptree.add((b,a))
    added = {a,b}
    unadded = VERTICES-added

    while unadded:
        a = random.choice(list(added))
        b = random.choice(list(unadded))
        sptree.add((a,b))
        #sptree.add((b,a))
        added.add(b)
        unadded.remove(b)

    return SpanningTree(sptree) # return the new SpanningTree

def performance(func):
    """
    Wrapper function to capture run time
    Input: a function (whose performance will be measured)
    Output: a function (returning run time, each generation's sptree lengths, and
                        the best result found by the algorithm)
    """
    def wrapper():
        start_time = time.time()
        gen_lengths, result = func()
        end_time = time.time()
        run_time = end_time - start_time
        return run_time, gen_lengths, result
    return wrapper

def get_init_pop():
    """
    Function to randomly generate an initial population of sptrees
    Input: None
    Output: a population (list) of sptrees (SpanningTree)
    """
    init_pop = [] # list for the population of sptrees
    for i in range(POP_SIZE): # generate as many new sptrees as we need
        new_sptree = get_rand_sptree() # get new randomly generated sptree
        # check the sptree is not a duplicate; save it as a SpanningTree in the pop
        if new_sptree not in init_pop: init_pop.append(new_sptree)
    return init_pop

def get_next_pop(pop):
    """
    Function to take a population and from it produce the next population
    The DISCARD variable determines the portion of the population replaced in
    each generation
    Input: a population (list) of sptrees (SpanningTree)
    Output: a population (list) of sptrees (SpanningTree)
    """
    next_pop = sorted(pop.copy(), key=attrgetter('length'))
    children = [] # for the children of the new generation

    # Get a child through wisdom of the crowds
    while (True):
        crowd_child = SpanningTree(get_crowd_wisdom(next_pop))
        logging.info("Crowd-Wisdom Kid: {}".format(crowd_child))
        if crowd_child not in next_pop:
            children.append(crowd_child)
            break # done, leave loop

    # Get children through genetic reproduction
    while len(children) < DISCARD: # make as many children as we need
        if (pop): # if there are valid parents, try and make children
            pop, kid1, kid2 = reproduce(pop)
            # Prevent duplicates from existing in the population
            if kid1 not in next_pop and kid1 not in children:
                children.append(kid1)
            else: logging.debug("kid1 is a duplicate")
            if kid2 not in next_pop and kid2 not in children:
                children.append(kid2)
            else: logging.debug("kid2 is a duplicate")
        else: # increase genetic diversity with a randomly generated sptree
            logging.debug("Adding random sptree to children")
            children.append(get_rand_sptree())

    children = sorted(children, key=attrgetter('length'))
    if mylevel == logging.DEBUG:
        print("Children:")
        for child in children: print(child)
    # Replace the pop's worst sptrees with children, if they are superior
    for i in range(DISCARD): # for the selected percentage of the population
        if children[0].length < next_pop[-1].length: # if the child is better
            logging.info("Discarding {}".format(next_pop[-1]))
            next_pop.pop(-1) # discard the worst from the old population
            next_pop.append(children.pop(0)) # replace with the best child
            next_pop = sorted(next_pop, key=attrgetter('length')) # re-sort
        else: break # leave the for loop if out of superior children

    return next_pop

def complete_tree(kid, parent):
    """
    Function to identify disjoint portions of a kid tree and select best edge(s) from
    the parent that will join these portions and complete the kid tree
    Input: incomplete kid sptree (set) and parent sptree (SpanningTree)
    Output: complete kid sptree (set)
    """
    logging.info("Completing tree...")

    # Determine all groups from the edges
    kid_copy = kid.copy() # save off
    groups = [set(edge) for edge in kid]
    newgroups = [groups.pop()] # prime
    logging.debug("first newgroup {}".format(newgroups))
    while groups: # for each group
        group = groups.pop() # get group
        logging.debug("group {}".format(group))
        for newgroup in newgroups: # for each newgroup
            # detect cycle-causing edges, vets crowd-wisdom selections
            if len(group)==2 and group.issubset(newgroup):
                logging.debug("Dropping invalid edge {}".format(group))
                edge = list(group) # to check both directions
                kid_copy.discard((edge[0],edge[1])) # prevent from being used
                kid_copy.discard((edge[1],edge[0]))
                break # drops the group
            if not group.isdisjoint(newgroup): # any of group is in newgroup
                logging.debug("Adding {} to {}".format(group,newgroup))
                group.update(newgroup) # incorporate newgroup
                groups.append(group) # store back in to check again
                newgroups.remove(newgroup) # not finalized yet
                break # done with this group
        else: # needs a newgroup
            logging.debug("Adding newgroup {}".format(group))
            newgroups.append(group)
    groups = newgroups # store back into groups
    logging.debug("Groups {}".format(groups))

    # Incorporate the unadded vertices as groups (of one)
    kid_unadded = VERTICES - {v for edge in groups for v in edge}
    logging.debug("kid_unadded {}".format(kid_unadded))
    for v in kid_unadded: groups.append({v})
    logging.info("All Groups {}".format(groups))

    # Select edge(s) to group disjointed portions together
    coregroup = groups.pop(-1) # prime with group
    logging.info("Coregroup {}".format(coregroup))
    while groups: # add best edge available from parent
        unadded = VERTICES - coregroup # available vertices
        logging.debug("unadded {}".format(unadded))
        valid_edges = [(a,b) for a in coregroup for b in unadded
                       if (a,b) in parent.sptree]
        #logging.debug("valid edges {}".format(valid_edges))
        if len(valid_edges)==0: raise Exception("Somehow no valid edges")
        # Add shortest edge available to grow/complete the tree
        valid_edges = sorted(valid_edges, key=lambda e: DISTS[e])
        logging.info("Appending edge {}".format(valid_edges[0]))
        kid_copy.add(valid_edges[0]) # add final edge
        for group in groups: # ID its group
            if valid_edges[0][1] in group:
                coregroup.update(group)
                logging.debug("Added group {}".format(group))
                groups.remove(group)
                break # found group, done
        else: raise Exception("Didn't find group for the edge")
        logging.debug("Coregroup {}".format(coregroup))

    if not len(kid_copy) == NUM_VERTICES-1:
        raise Exception("Wrong number of edges")

    logging.debug("Kid: {}".format(kid_copy))
    return kid_copy

def get_fitness(pop):
    """
    Function to calculate the fitness ratios of all sptrees in a population
    Input: a population (list) of sptrees (SpanningTree)
    Output: a list of probabilities corresponding to the input population
    """
    logging.info("Evaluating population...")
    xmax = max([x.length for x in pop]) # ID the max sptree length
    xmin = min([x.length for x in pop]) # ID the min sptree length
    if xmax==xmin: return [1/len(pop) for x in pop] # all sptrees are the same

    fitness = [1-((x.length-xmin)/(xmax-xmin)) for x in pop] # initial scale
    fitness = [f/sum(fitness) for f in fitness] # adjust so sums up to 1
    logging.debug("Sum of fitnesses 1?: {}".format(sum(fitness)))
    logging.debug("Population fitnesses: {}".format(fitness))
    return fitness

def crossover(mom, dad):
    """
    Function performing the crossover (and potential mutation) of "genes" from
    two parents to make two children
    Input: two parent sptrees (SpanningTree)
    Output: two child sptrees (SpanningTree)
    """
    logging.info("Making children...")
    # randomly select crossover point (excluding invalid indices 0 and -1)
    #cross = random.randrange(1,NUM_VERTICES-1)
    # crossover approximately 1/3 of the edges
    cross = (NUM_VERTICES-1)//3*2
    logging.debug("Crossover point: {}".format(cross))

    # Use best mom and dad edges in the children
    mom_edges = [e for e in mom.sptree if e[0]<e[1]] # pull unique edges
    dad_edges = [e for e in dad.sptree if e[0]<e[1]] # pull unique edges
    mom_edges = sorted(mom_edges, key=lambda e: DISTS[e]) # sort by edge lengths
    dad_edges = sorted(dad_edges, key=lambda e: DISTS[e]) # sort by edge lengths

    kid1 = {mom_edges[i] for i in range(cross)} # start with "genes" from mom
    logging.debug("kid1 from mom {}".format(kid1))
    kid1 = complete_tree(kid1, dad) # complete kid1 with "genes" from dad
    kid1 = SpanningTree(kid1) # make into SpanningTree
    kid1.mutate() # check/execute mutation
    logging.info("Kid 1: {}".format(kid1))

    kid2 = {dad_edges[i] for i in range(cross)} # start with "genes" from dad
    logging.debug("kid2 from dad {}".format(kid2))
    kid2 = complete_tree(kid2, mom) # complete kid2 with "genes" from mom
    kid2 = SpanningTree(kid2) # make into SpanningTree
    kid2.mutate() # check/execute mutation
    logging.info("Kid 2: {}".format(kid2))
    return kid1, kid2

def reproduce(pop):
    """
    Function to select parents according to their fitness within the population
    and get their children
    Input: a population (list) of sptrees (SpanningTree)
    Output: the remaining population (list) minus the mom and dad,
            kid1 (SpanningTree), and kid2 (SpanningTree)
    """
    logging.info("Reproducing children genetically...")
    # If crossover hits, make new children
    test = random.random()
    logging.debug("Crossover test: {}".format(test))
    if test <= SpanningTree.pc:
        logging.debug("Using reproduction with crossover...")
        popfit = get_fitness(pop)
        if (len(popfit)) > 2:
            # Using numpy's random.choice, select 2, without replacement, from the
            # population with probabilities popfit
            mom, dad = nprandom.choice(len(popfit), size=2, replace=False, p=popfit)
            logging.debug("Mom, Dad indices: {}, {}".format(mom, dad))
            # Get the SpanningTrees themselves, removing them from the reproduction pool
            # Pop higher index first to ensure correct, valid sptrees are popped
            if mom > dad: mom, dad = pop.pop(mom), pop.pop(dad)
            else:         dad, mom = pop.pop(dad), pop.pop(mom)
        else: mom, dad = pop.pop(), pop.pop()
        logging.info("Mom: {}".format(mom))
        logging.info("Dad: {}".format(dad))

        # Execute crossover
        kid1, kid2 = crossover(mom, dad)
        return pop, kid1, kid2

    # When crossover does not hit, return two randomly generated sptrees
    else:
        logging.debug("Using random sptrees as children...")
        kid1, kid2 = get_rand_sptree(), get_rand_sptree()
        logging.info("Kid 1: {}".format(kid1))
        logging.info("Kid 2: {}".format(kid2))
        return pop, kid1, kid2

def get_edge_count(pop):
    """
    Function to add up every instance of an edge in a population of sptrees
    Input: a population (list) of sptrees (SpanningTree)
    Output: a list of percentages corresponding to the input population
    """
    logging.info("Counting edges...")
    edge_count = [[0 for i in range(NUM_VERTICES)] for j in range(NUM_VERTICES)]
    for x in pop: # for each sptree
        # for each unique edges only; no need to "mirror" matrix
        for edge in {e for e in x.sptree if e[0]<e[1]}:
            edge_count[edge[0]-1][edge[1]-1] +=1
    logging.debug("Edge Counts: {}".format(edge_count))

    # Convert all counts to percentages
    for i in range(NUM_VERTICES):
        # only need to review "top half" of matrix
        for j in range(i+1,NUM_VERTICES):
            # divide count by number of edges per sptree x population size
            edge_count[i][j] = edge_count[i][j]/((NUM_VERTICES-1)*POP_SIZE)

    return edge_count

def get_crowd_wisdom(pop):
    """
    Function to apply the Wisdom of Crowds to a population of sptrees
    Input: a population (list) of sptrees (SpanningTree)
    Output: a new kid sptree (SpanningTree)
    """
    logging.info("Applying Wisdom of the Crowds...")
    # First select some edges based on Crowd Wisdom
    edge_count = get_edge_count(pop) # get edge count matrix
    flat_list = [item for sublist in edge_count for item in sublist] # "flatten"
    # pick as many edges as determined by CROWD, based on agreement percentages
    flat_inds = list(nprandom.choice(NUM_VERTICES**2, size=CROWD, replace=False,
                                     p=flat_list)) # returns indices
    edge_inds = [divmod(ind, NUM_VERTICES) for ind in flat_inds] # list of tuples
    kid = {(x[0]+1, x[1]+1) for x in edge_inds} # convert to vertex nums
    logging.debug("Selected Edges: {}".format(kid)) # a.k.a. the kid so far

    # Evaluate edges for validity i.e. detect cycles

    # Complete kid with greedy edge choices
    parent = {(a,b) for a in VERTICES for b in range(a+1,NUM_VERTICES+1)}
    parent = SpanningTree(parent) # every possible edge is an option
    kid = complete_tree(kid, parent) # complete with "parent" that is all edges

    return kid

@performance
def algorithm():
    """
    Function to make all the generations according to a Genetic Algorithm with a
    Wisdom of the Crowds approach
    Input: None
    Output: list of each generation's sptree lengths (list) for use in
            performance statistics of the algorithm, result (SpanningTree) that is the
            very best sptree found at the completion of the algorithm
    """
    gen_lengths = [] # list of sets of each generation's sptree lengths
    for i in range(NUM_GENS):
        logging.info("Getting population {}...".format(i))
        if i==0: mypop = get_init_pop()
        else:    mypop = get_next_pop(mypop)

        mypop = sorted(mypop, key=attrgetter('length'))
        if mylevel == logging.DEBUG:
            print("Population:")
            for sptree in mypop: print(sptree)
        else: # mylevel >= logging.INFO:
            print("Best of Population {}: {}".format(i,mypop[0]))

        gen_lengths.append([r.length for r in mypop])

    result = mypop[0]
    return gen_lengths, result

################################# MAIN PROGRAM ##################################

print("Welcome to the Shortest Total-Path-Length Spanning Tree (SPST) Problem")

# Specify desired logging statements
mylevel = logging.WARNING
logging.basicConfig(level=mylevel, format='%(levelname)s: %(message)s')

# Define global variables
COORDS = read_in_coords("mvhRandom80.tsp") # dict: {key= vertex num: value= (x,y)]
NUM_VERTICES = len(COORDS) # for easy use of the number of vertices in this SPST
VERTICES = set(range(1,NUM_VERTICES+1)) # all the vertex numbers in a set
##if mylevel == logging.DEBUG:
##    print("Printing vertex coordinates...")
##    print(COORDS)

DISTS = calc_dists() # dictionary: {key= vertex pair: value= distance}
##if mylevel == logging.DEBUG:
##    print("Printing distances between the vertices...")
##    print(DISTS)

#POP_SIZE = NUM_VERTICES//2//2*2 # population size of a smaller number
#POP_SIZE = NUM_VERTICES//2*2 # population size of as many sptrees as vertices
POP_SIZE = 40
NUM_GENS = 30 # the number of generations each run should have

DISCARD = POP_SIZE//3//2*2 # will replace ~1/3 of each generation's population
CROWD = NUM_VERTICES//3*2 # number of edges to use based on crowd opinion

# Execute the algorithm, with @performance capturing desired statistics
runtime, gen_lengths, result = algorithm()
print("Algorithm runtime: {:.5f} seconds".format(runtime))
print("Best Result: {}".format(result))

# Process the generation lengths for meaningful statistics
mins, maxs, avgs = [], [], []
for i in gen_lengths:
    mins.append(min(i))
    maxs.append(max(i))
    avgs.append(sum(i)/len(i))

################################ PLOTTING GRAPHS ################################

logging.info("Plotting statistical data...")
plt.figure()
plt.xlabel('Generations')
plt.ylabel('Distance')
plt.title('GA WoC SPST Performance')
plt.plot(mins,'g') # Mininum distances plotted with green line
plt.plot(maxs,'r') # Maximum distances plotted with red line
plt.plot(avgs,'b') # Average distances plotted with blue line
plt.show()

############################### GRAPHICS DISPLAY ################################

def get_disp_coords():
    """
    Function will scale the vertex coordinates to the window size defined by
    my_height and my_width and it flips the y values to accomodate pygame's
    "flipped" y axis
    """
    disp_coords = {} # coordinates for graphics rendering

    # Get max and min x and y vertex coordinates
    xs = {coord[0] for coord in COORDS.values()}
    ys = {coord[1] for coord in COORDS.values()}

    # Scale coordinates to fit window defined by WINDOW_WIDTH and WINDOW_HEIGHT
    # Linear translation and scaling: f(x) = ((b-a)(x-min)/(max-min))+a
    temp_x = temp_y = 0
    for i,c in COORDS.items():
        # a is 25 and b is WINDOW_WIDTH-25 to build in a 25pt border
        temp_x = ((WINDOW_WIDTH-25-25)*(c[0]-min(xs))/(max(xs)-min(xs)))+25
        # a is 25 and b is WINDOW_HEIGHT-25 to build in a 25pt border
        temp_y = ((WINDOW_HEIGHT-25-25)*(c[1]-min(ys))/(max(ys)-min(ys)))+25
        temp_y = WINDOW_HEIGHT-temp_y # "flip" the y axis
        disp_coords[i] = (temp_x,temp_y)

    return disp_coords

def draw_sptree(mysptree):
    """
    Function to draw every line in the best sptree
    """
    sptree = [e for e in mysptree.sptree if e[0]<e[1]] # get unique edges
    for edge in sptree:
        fm_x,fm_y = DISP_COORDS[edge[0]]
        to_x,to_y = DISP_COORDS[edge[1]]
        # Draw the sptree lines using the fm and to (x,y) in blue
        pygame.draw.line(screen, black, [fm_x,fm_y], [to_x,to_y], 3)

    return

logging.info("Drawing the best Spanning Tree...")
# Initialize the game engine
pygame.init()

# Define basic colors; follows RGB scheme
black, white = (0, 0, 0),(255, 255, 255)

# Set width and height of window
WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 900

# Set to be windowed at the given size
screen=pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])

# Set window title
pygame.display.set_caption("Genetic & Wisdom of the Crowds Shortest " \
                           "Total-Path-Length Spanning Tree by Madeline Hundley")

clock = pygame.time.Clock() # Manage how fast the screen updates
# Pick font and size; necessary to distribute program to other users
font = pygame.font.Font("C:/Windows/Fonts/times.ttf", 14)

# Get the display coordinates for the vertices
DISP_COORDS = get_disp_coords()

done = False # Loop until user clicks close
while not done:
    # Event Processing:
    for event in pygame.event.get(): # User did something
        if event.type == pygame.QUIT: # If user clicks close
            done=True

    # Drawn objects stack on top of each other; draw from "bottom" to "top"
    screen.fill(white) # create white background
    draw_sptree(result) # draw every edge in the sptree

    # Render all vertices and label them:
    for i,c in DISP_COORDS.items():
        pygame.draw.rect(screen, white, [c[0]-12,c[1]-12,22,20])
        pygame.draw.rect(screen, black, [c[0]-12,c[1]-12,22,20], 3)
        vertex_name = font.render(str(i), True, black)
        if len(str(i))>1: screen.blit(vertex_name, [c[0]-8,c[1]-10])
        else:             screen.blit(vertex_name, [c[0]-4,c[1]-10])

    pygame.display.flip() # Push drawings to screen
    clock.tick(20) # limits loop to 20 fps

# Close window neatly
pygame.quit()
