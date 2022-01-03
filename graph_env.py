import math
import random
import time

import matplotlib
import numpy as np
import networkx as nx
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy


#from numba.cuda import grid

grid_points = []
edge_keys = {}

class GraphEnv:
    def __init__(self, points, neighbor_distance, home_nodes):
        global grid_points
        global edge_keys

        self.currentPlayer = 1
        #self.graph = graph
        #self.pos = pos
        self.home_nodes = home_nodes
        grid_points = points
        edge_keys = self.find_neighbors(neighbor_distance)
#        self.gameState = GameState(np.array(
#            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#             0, 0, 0, 0, 0, 0], dtype=np.int), 1)
        self.graph_state = GraphState(replaceArray(np.zeros(len(edge_keys))), home_nodes, [], 1)
        #self.actionSpace = np.zeros(len(points) * len(points))
        self.actionSpace = np.zeros(len(edge_keys))
        #self.pieces = {'1': 'X', '0': '-', '-1': 'O'}
        self.grid_shape = (len(edge_keys), 1)
        self.input_shape = (2, len(edge_keys), 1)
        self.name = 'graph'
        self.state_size = len(self.graph_state.binary)
        self.action_size = len(self.actionSpace)


    def reset(self):
        GraphState(replaceArray(np.zeros(len(edge_keys))), self.home_nodes, [], 1)
        self.currentPlayer = 1
        return self.graph_state

    def step(self, action):
        next_state, value, done = self.graph_state.takeAction(action)
        self.graph_state = next_state
        self.currentPlayer = -self.currentPlayer
        info = None
        return ((next_state, value, done, info))


    def identities(self, state, actionValues):
        identities = [(state, actionValues)]

        currentBoard = state.edges
        currentAV = actionValues

        identities.append((GraphState(currentBoard, self.home_nodes, [], state.playerTurn), currentAV))
        return identities


    def find_neighbors(self, distance):
        i = 0
        for point in grid_points:
            neighbor_points = find_neighboring_nodes(point, distance, distance)
            for neighbor_point in neighbor_points:
                if len(edge_keys) == 0:
                    edge_keys[(point, neighbor_point)] = i
                    i = i + 1
                elif (point, neighbor_point) not in edge_keys and (neighbor_point, point) not in edge_keys:
                    edge_keys[(point, neighbor_point)] = i
                    #edge_keys[(neighbor_point, point)] = i
                    i = i + 1

        return edge_keys


def calculate_max_allowed_cost(self):
    # TODO: to be implemented
    return 0;


def calculate_cost(self):
    # TODO: to be implemented
    return 1;


def find_neighboring_nodes(grid_node, dist_x, dist_y):
    neighbors = []
    for node in grid_points:
        #if abs(node[0] - grid_node[0]) == dist_x and abs(node[1] - grid_node[1]) == dist_y:
        distance = calc_distance(node, grid_node)
        if distance > 0 and distance < 1.05 * dist_x:
            neighbors.append(node)
    return neighbors


def calc_distance(node1, node2):
    return abs(((node2[0] - node1[0]) ** 2 + (node2[1] - node1[1]) ** 2) ** 0.5)


def replaceArray(edges):
    for i in range(len(edges)):
        if edges[i] == 0:
            edges[i] = 1
        else:
            edges[i] = 0
    return edges


class GraphState:
    def __init__(self, edges, home_nodes, possible_connect_nodes, playerTurn):
        #self.edges = edges
        self.home_nodes = home_nodes
        #self.pieces = {'1': 'X', '0': '-', '-1': 'O'}
        self.edges = edges
        self.edge_matrix = self.create_matrix()
        self.winners = []
        self.possible_connect_nodes = []
        self.requiredEdges = []
        self.playerTurn = playerTurn
        self.binary = self._binary()
        self.id = self._convertStateToId()
        self.allowedActions = self._allowedActions()
        self.isEndGame = self._checkForEndGame()
        self.value = self._getValue()
        self.score = self._getScore()
        #self.max_allowed_cost = calculate_max_allowed_cost(self.graph)
        #self.cost = calculate_cost(self.graph)


    def create_matrix(self):
        matrix = np.zeros((len(grid_points), len(grid_points)))
        non_zero_indices = np.nonzero(self.edges)
        for i in list(non_zero_indices[0]):
            edge = list(edge_keys.keys())[list(edge_keys.values()).index(i)]
            node1 = edge[0]
            node2 = edge[1]
            node1_index = grid_points.index(node1)
            node2_index = grid_points.index(node2)
            matrix[node1_index][node2_index] = 1
            matrix[node2_index][node1_index] = 1
        return matrix


    def _binary(self):

        currentplayer_position = self.edges
        currentplayer_position[self.edges == self.playerTurn] = 1

        other_position = self.edges
        #other_position[self.edges == -self.playerTurn] = 1
        other_position[self.edges == self.playerTurn] = 1

        position = np.append(currentplayer_position, other_position)

        return (position)

    def _convertStateToId(self):
        player1_position = self.edges
        player1_position[self.edges == 1] = 1

        other_position = self.edges
        #other_position[self.edges == -1] = 1
        other_position[self.edges == 1] = 1

        position = np.append(player1_position, other_position)

        id = ''.join(map(str, position))

        return id


    def _checkForEndGame(self):
        if self.check_edge_costs2() == 0:
            '''
            matrix = np.zeros((len(grid_points), len(grid_points)))
            non_zero_indices = np.nonzero(self.edges)
            nodes_with_edges = set()
            for idx in list(non_zero_indices[0]):
                edge = list(edge_keys.keys())[list(edge_keys.values()).index(idx)]
                node1 = edge[0]
                node2 = edge[1]
                node1_index = grid_points.index(node1)
                node2_index = grid_points.index(node2)
                matrix[node1_index][node2_index] = 1
                matrix[node2_index][node1_index] = 1
                nodes_with_edges.add(node1)
                nodes_with_edges.add(node2)
            G = nx.from_numpy_matrix(matrix)
            descendents = nx.algorithms.descendants(G, grid_points.index(self.home_nodes[0]))
            descendents.add(grid_points.index(self.home_nodes[0]))
            descendent_grid_points = []
            for point in descendents:
                descendent_grid_points.append(grid_points[point])
            if len(nodes_with_edges) > 0 and all(item in descendent_grid_points for item in list(nodes_with_edges)):
                pos = {}
                for node in G.nodes:
                    pos[node] = grid_points[node]
                    val_map = {}
                    values = []
                    if pos[node] in self.home_nodes:
                        values.append(1)
                    else:
                        values.append(.25)
                f = plt.figure()
                nx.draw(G, pos, edge_color="lightgray", node_size=1, with_labels=False)
                f.savefig("graph" + str(time.strftime("%Y%m%d-%H%M%S")) + ".png")
                '''
            return 1
            #else:
            #    return 0
        return 0

    def _getValue(self):
        # This is the value of the state for the current player
        # i.e. if the previous player played a winning move, you lose
        #for x, y, z, a in self.winners:
        #    if (self.graph[x] + self.graph[y] + self.graph[z] + self.graph[a] == 4 * -self.playerTurn):
        #        return (-1, -1, 1)

        # check metasquares/game.py implementation

        currentPlayer_cost = sum(self.edges)
        oppositionPlayer_cost = sum(self.edges)

        #if self.check_edge_costs() == 1:
        #    return (-1, -1, 1)

        if currentPlayer_cost < oppositionPlayer_cost:
            return (1, currentPlayer_cost, oppositionPlayer_cost)
        #elif currentPlayer_cost > oppositionPlayer_cost:
        else:
            return (-1, currentPlayer_cost, oppositionPlayer_cost)
        #else:
        #    return (0, currentPlayer_cost, oppositionPlayer_cost)

    def _getScore(self):
        tmp = self.value
        return (tmp[1], tmp[2])


    def takeAction(self, action):
        newBoard = np.array(self.edges)
        nodes = []

        action = [action]
        '''
        if -1 in action:
            for action_inst in action:
                if action_inst in edge_keys.values():
                    action = [action_inst]
                    break
        '''
        for edge_id in action:
            edge = list(edge_keys.keys())[list(edge_keys.values()).index(edge_id)]
            node1 = edge[0]
            node2 = edge[1]
            #node1_index = grid_points.index(node1)
            #node2_index = grid_points.index(node2)
            newBoard[edge_id] = 0

            if node1 not in self.home_nodes:
                nodes.append(node1)
            if node2 not in self.home_nodes:
                nodes.append(node2)


        #for node in nodes:
        #    if node not in newState.home_nodes:
        #        newState.possible_connect_nodes.append(node)

        # newState = GraphState(newBoard, self.home_nodes, -self.playerTurn)
        newState = GraphState(newBoard, self.home_nodes, nodes, self.playerTurn)

        value = 0
        done = 0

        if newState.isEndGame:
            value = newState.value[0]
            done = 1
        return (newState, value, done)

    def render(self, logger):
        #for r in range(6):
        #    logger.info([self.pieces[str(x)] for x in self.graph[7 * r: (7 * r + 7)]])
        logger.info('--------------')


    def check_edge_costs(self):
        matrix = np.zeros((len(grid_points), len(grid_points)))
        non_zero_indices = np.nonzero(self.edges)
        # if sum(non_zero_indices) > 0:
        for idx in list(non_zero_indices[0]):
            edge = list(edge_keys.keys())[list(edge_keys.values()).index(idx)]
            node1 = edge[0]
            node2 = edge[1]
            node1_index = grid_points.index(node1)
            node2_index = grid_points.index(node2)
            matrix[node1_index][node2_index] = 1
            matrix[node2_index][node1_index] = 1
        G = nx.from_numpy_matrix(matrix)

        node_degrees = G.degree()
        nodes = [i[0] for i in node_degrees]
        degrees = [i[1] for i in node_degrees]

        if sum(i > 2 for i in degrees) > len(self.home_nodes):
            return 1
        '''
        else:
            for home_node in self.home_nodes:
                if degrees[nodes.index(grid_points.index(home_node))] != 1:
                    return -1
        '''
        return 0



    def check_for_paths_between_homenodes(self):
        matrix = np.zeros((len(grid_points), len(grid_points)))
        non_zero_indices = np.nonzero(self.edges)
        #if sum(non_zero_indices) > 0:
        for idx in list(non_zero_indices[0]):
            edge = list(edge_keys.keys())[list(edge_keys.values()).index(idx)]
            node1 = edge[0]
            node2 = edge[1]
            node1_index = grid_points.index(node1)
            node2_index = grid_points.index(node2)
            matrix[node1_index][node2_index] = 1
            matrix[node2_index][node1_index] = 1

            #matrix = np.reshape(copy.deepcopy(self.edges), newshape=(len(grid_points), len(grid_points)))
        G = nx.from_numpy_matrix(matrix)
        # replace with home_nodes instead of hardcoding 20
        descendents = nx.algorithms.descendants(G, grid_points.index(self.home_nodes[0]))
        descendents.add(grid_points.index(self.home_nodes[0]))
        descendent_grid_points = []
        for point in descendents:
            descendent_grid_points.append(grid_points[point])
        # all(elem in descendents for elem in self.home_nodes)
        if all(item in descendent_grid_points for item in self.home_nodes):
            #nx.draw(G, pos, node_color=values, edge_color="lightgray", node_size=1, with_labels=False)
            #plt.show()
            return True
        return False


    def _allowedActions(self):
        allowedActions = []
        non_zero_indices = np.nonzero(self.edges)[0]
        random.shuffle(non_zero_indices)
        for idx in non_zero_indices:
            #edges = copy.deepcopy(self.edges)
            matrix = copy.deepcopy(self.edge_matrix)
            #edges[idx] = 0
            edge = list(edge_keys.keys())[list(edge_keys.values()).index(idx)]
            node1 = edge[0]
            node2 = edge[1]
            node1_index = grid_points.index(node1)
            node2_index = grid_points.index(node2)
            matrix[node1_index][node2_index] = 0
            matrix[node2_index][node1_index] = 0
            G = nx.from_numpy_matrix(matrix)
            # replace with home_nodes instead of hardcoding 20
            descendents = nx.algorithms.descendants(G, grid_points.index(self.home_nodes[0]))
            descendents.add(grid_points.index(self.home_nodes[0]))
            descendent_grid_points = []
            for point in descendents:
                descendent_grid_points.append(grid_points[point])
            # all(elem in descendents for elem in self.home_nodes)
            if all(item in descendent_grid_points for item in self.home_nodes) and idx not in allowedActions:
                allowedActions.append(idx)
                if len(allowedActions) == 10:
                    return allowedActions
        return allowedActions


    def _allowedActions3(self):
        allowedActions = []
        actions = []
        edges = copy.deepcopy(self.edges)

        non_zero_indices = np.nonzero(edges)[0]
        random.shuffle(non_zero_indices)

        for idx in non_zero_indices:
            if len(allowedActions) == 2:
                return allowedActions

            matrix = copy.deepcopy(self.edge_matrix)
            edges[idx] = 0
            edge = list(edge_keys.keys())[list(edge_keys.values()).index(idx)]
            node1 = edge[0]
            node2 = edge[1]
            node1_index = grid_points.index(node1)
            node2_index = grid_points.index(node2)
            matrix[node1_index][node2_index] = 0
            matrix[node2_index][node1_index] = 0
            G = nx.from_numpy_matrix(matrix)
            # replace with home_nodes instead of hardcoding 20
            descendents = nx.algorithms.descendants(G, grid_points.index(self.home_nodes[0]))
            descendents.add(grid_points.index(self.home_nodes[0]))
            descendent_grid_points = []
            for point in descendents:
                descendent_grid_points.append(grid_points[point])
            # all(elem in descendents for elem in self.home_nodes)
            if all(item in descendent_grid_points for item in self.home_nodes) and idx not in (item for sublist in
                                                                                               allowedActions for item
                                                                                               in sublist):
                actions.append(idx)
                if len(actions) == 10:
                    allowedActions.append(actions)
                    actions = []
                    edges = copy.deepcopy(self.edges)

        if len(actions) > 0:
            allowedActions.append(actions)
        return allowedActions



    def _allowedActions2(self):
        allowedActions = []
        actions = []
        #random.shuffle(edges)
        edges = copy.deepcopy(self.edges)
        for idx in range(len(self.edges)):
            if len(allowedActions) == 2:
                return allowedActions
            if self.edges[idx] == 1:
                edges[idx] = 0
                matrix = np.zeros((len(grid_points), len(grid_points)))
                non_zero_indices = np.nonzero(edges)
                # if sum(non_zero_indices) > 0:
                for i in list(non_zero_indices[0]):
                    edge = list(edge_keys.keys())[list(edge_keys.values()).index(i)]
                    node1 = edge[0]
                    node2 = edge[1]
                    node1_index = grid_points.index(node1)
                    node2_index = grid_points.index(node2)
                    matrix[node1_index][node2_index] = 1
                    matrix[node2_index][node1_index] = 1

                    # matrix = np.reshape(copy.deepcopy(self.edges), newshape=(len(grid_points), len(grid_points)))
                G = nx.from_numpy_matrix(matrix)
                # replace with home_nodes instead of hardcoding 20
                descendents = nx.algorithms.descendants(G, grid_points.index(self.home_nodes[0]))
                descendents.add(grid_points.index(self.home_nodes[0]))
                descendent_grid_points = []
                for point in descendents:
                    descendent_grid_points.append(grid_points[point])
                # all(elem in descendents for elem in self.home_nodes)
                if all(item in descendent_grid_points for item in self.home_nodes) and idx not in (item for sublist in allowedActions for item in sublist):
                    actions.append(idx)
                    if len(actions) == 10:
                        allowedActions.append(actions)
                        actions = []
                        edges = copy.deepcopy(self.edges)

        if len(actions) > 0:
            allowedActions.append(actions)
        return allowedActions


    def check_edge_costs2(self):
        edges = copy.deepcopy(self.edges)
        non_zero_indices = np.nonzero(edges)[0]
        for idx in non_zero_indices:
            matrix = copy.deepcopy(self.edge_matrix)
            edges[idx] = 0
            edge = list(edge_keys.keys())[list(edge_keys.values()).index(idx)]
            node1 = edge[0]
            node2 = edge[1]
            node1_index = grid_points.index(node1)
            node2_index = grid_points.index(node2)
            matrix[node1_index][node2_index] = 0
            matrix[node2_index][node1_index] = 0
            G = nx.from_numpy_matrix(matrix)
            descendents = nx.algorithms.descendants(G, grid_points.index(self.home_nodes[0]))
            descendents.add(grid_points.index(self.home_nodes[0]))
            descendent_grid_points = []
            for point in descendents:
                descendent_grid_points.append(grid_points[point])
            if all(item in descendent_grid_points for item in self.home_nodes):
                return 1
        return 0
