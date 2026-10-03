class ListNode:
    def __init__(self,key=0,value=0):
        self.key=key
        self.val=value
        self.next=None
        self.prev=None

class LRUCache:
    def __init__(self,capacity):
        self.capacity=capacity
        self.hashmap={}
        self.head=ListNode()
        self.tail=ListNode()
        self.head.next=self.tail
        self.tail.prev=self.head
    
    def remove_node(self,node):
        node.prev.next=node.next
        node.next.prev=node.prev

    def add_to_head(self,node):
        node.next=self.head.next
        node.prev=self.head
        self.head.next.prev=node
        self.head.next=node

    def move_node_to_head(self,node):
        self.remove_node(node)
        self.add_to_head(node)

    def get(self,key):
        if key not in self.hashmap:
            return -1
        else:
            node=self.hashmap[key]
            self.move_node_to_head(node)
            return node.val

    def put(self,key,value):
        if key in self.hashmap:
            node=self.hashmap[key]
            node.val=value
            self.move_node_to_head(node)
        else:
            if len(self.hashmap)==self.capacity:
                del self.hashmap[self.tail.prev.key]
                self.remove_node(self.tail.prev)
            node=ListNode(key,value)
            self.hashmap[key]=node
            self.add_to_head(node)

if __name__ == "__main__":
    c = LRUCache(2)
    c.put(1, 1); c.put(2, 2)
    assert c.get(1) == 1            # ① 最近使用：返回 1
    c.put(3, 3)                     # ② 满了 → 淘汰最久未用(2)
    assert c.get(2) == -1           # 2 应已被淘汰
    assert c.get(3) == 3
    c.put(4, 4)                     # ③ 满了 → 淘汰最久未用(1)
    assert c.get(1) == -1           # 1 应已被淘汰
    assert c.get(3) == 3 and c.get(4) == 4
    print("全部通过 ✅")







