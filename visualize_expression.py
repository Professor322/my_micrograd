import graphviz
import Value from value.py

# initially this function was supposed to print graph
# without graphviz, that is why BFS is used here
# but after adding graphviz visualization there is no 
# difference which traversal to use
# all this complexity to avoid duplciation and make the same variale represented correctly
# in different expressions
# TODO: separate concerns: 1) traverse graph to collect nodes and edges
#                          2) build graph visualization
def ValueBFS(value):
    assert isinstance(value, Value), "Incorrect argument"

    dag = graphviz.Digraph(comment = "Expression tree")
    # store in list (node, parent_id)
    list = [(value, -1)]
    visited_nodes_dict = {}
    visited_edges = set()
    node_id = 0
    while (len(list)):
        fixed_len = len(list)
        for _ in range(0, fixed_len):
            curr_value, parent_id = list.pop(0)
            
            # create a node with unique identifier
            curr_value_id = node_id; node_id += 1
            op_id = -1
            # create value node and store in the dictionary for the lookups
            if not (curr_value in visited_nodes_dict): 
                visited_nodes_dict[curr_value] = (curr_value_id, -1)
                dag.attr('node', shape = 'box')
                dag.node(str(curr_value_id), 
                         f'data={curr_value.data:.3f} | grad={curr_value.grad:.3f}')
            else:
                curr_value_id, op_id = visited_nodes_dict.get(curr_value)
            
            # create op node if possible and creates an edge between op and value
            if curr_value._op != '':
                if op_id == -1:
                    op_id = node_id; node_id += 1
                    visited_nodes_dict[curr_value] = (curr_value_id, op_id)

                dag.attr('node', shape = 'ellipse')
                dag.node(str(op_id), curr_value._op)
                if not (op_id, curr_value_id) in visited_edges:
                    dag.edge(str(op_id), str(curr_value_id))
                    visited_edges.add((op_id, curr_value_id))
            

            if parent_id != -1:
                dag.edge(str(curr_value_id), str(parent_id))
                visited_edges.add((curr_value_id, parent_id))

            new_parent_id = op_id if op_id != -1 else curr_value_id 
            for child in curr_value._children:
                if child in visited_nodes_dict:
                    v, op = visited_nodes_dict.get(child)
                    u = new_parent_id
                    if (v, u) in visited_edges:
                        continue
                list.append((child, new_parent_id))
            
    display(dag)


