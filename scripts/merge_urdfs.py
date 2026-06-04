import xml.etree.ElementTree as ET
import os

def merge_urdfs(base_urdf, right_hand_urdf_path, left_hand_urdf_path, output_urdf):
    print(f"Loading Base: {base_urdf}...")
    tree_base = ET.parse(base_urdf)
    root_base = tree_base.getroot()

    attachment_links = ['right_hand', 'left_hand']

    # --- 1. Map all joints to build a parent -> child relationship tree ---
    joint_tree = {}
    for joint in root_base.findall('joint'):
        parent_elem = joint.find('parent')
        child_elem = joint.find('child')
        if parent_elem is not None and child_elem is not None:
            parent = parent_elem.attrib.get('link')
            child = child_elem.attrib.get('link')
            if parent not in joint_tree:
                joint_tree[parent] = []
            joint_tree[parent].append((child, joint))

    links_to_remove = set()
    joints_to_remove = set()

    def find_descendants(link_name):
        if link_name in joint_tree:
            for child_link, joint_elem in joint_tree[link_name]:
                links_to_remove.add(child_link)
                joints_to_remove.add(joint_elem)
                find_descendants(child_link)

    # --- 2. Find all descendants of the attachment points ---
    for attach_link in attachment_links:
        find_descendants(attach_link)

    # --- 3. Remove the identified child links and joints ---
    if links_to_remove or joints_to_remove:
        print(f"Removing {len(links_to_remove)} downstream links and {len(joints_to_remove)} downstream joints...")
        
    for link in root_base.findall('link'):
        if link.attrib.get('name') in links_to_remove:
            root_base.remove(link)
            
    for joint in root_base.findall('joint'):
        if joint in joints_to_remove:
            root_base.remove(joint)

    # --- 4. Strip visuals/collisions from the attachment links themselves ---
    for link in root_base.findall('link'):
        if link.attrib.get('name') in attachment_links:
            for tag in ['visual', 'collision']:
                for sub_elem in link.findall(tag):
                    link.remove(sub_elem)
    
    def append_hand(hand_path, parent_link, prefix, joint_name, rpy_offset="0 0 0"):
        print(f"Attaching {hand_path} to {parent_link} with RPY {rpy_offset}...")
        tree_hand = ET.parse(hand_path)
        root_hand = tree_hand.getroot()
        
        # Dynamically find the root link of the hand URDF
        hand_links = set(link.attrib.get('name') for link in root_hand.findall('link') if link.attrib.get('name'))
        hand_child_links = set(joint.find('child').attrib.get('link') for joint in root_hand.findall('joint') if joint.find('child') is not None)
        
        root_candidates = list(hand_links - hand_child_links)
        if not root_candidates:
            raise ValueError(f"Could not find a root link in {hand_path}")
        hand_root_link = root_candidates[0]
        
        hand_dir = os.path.dirname(os.path.abspath(hand_path))
        
        # --- 5. Prefix Names & Fix Mesh Paths ---
        for elem in root_hand:
            if elem.tag == 'link':
                old_name = elem.attrib.get('name')
                if old_name:
                    elem.attrib['name'] = prefix + old_name
                    
                for mesh in elem.findall('.//mesh'):
                    filename = mesh.attrib.get('filename')
                    if filename and not filename.startswith('package://') and not os.path.isabs(filename):
                        abs_path = os.path.normpath(os.path.join(hand_dir, filename))
                        mesh.attrib['filename'] = abs_path

            elif elem.tag == 'joint':
                old_name = elem.attrib.get('name')
                if old_name:
                    elem.attrib['name'] = prefix + old_name
                    
                parent_elem = elem.find('parent')
                if parent_elem is not None and parent_elem.attrib.get('link'):
                    parent_elem.attrib['link'] = prefix + parent_elem.attrib.get('link')
                    
                child_elem = elem.find('child')
                if child_elem is not None and child_elem.attrib.get('link'):
                    child_elem.attrib['link'] = prefix + child_elem.attrib.get('link')
        
        # --- 6. Append to Base URDF ---
        for elem in root_hand:
            if elem.tag in ['link', 'joint', 'material']:
                if elem.tag == 'material':
                    mat_name = elem.attrib.get('name')
                    existing_mats = [m.attrib.get('name') for m in root_base.findall('material')]
                    if mat_name in existing_mats and mat_name != "":
                        continue
                root_base.append(elem)
                
        # --- 7. Create the Welding Joint with Rotation ---
        fixed_joint = ET.Element('joint', attrib={'name': joint_name, 'type': 'fixed'})
        
        ET.SubElement(fixed_joint, 'origin', attrib={'xyz': '0 0 0', 'rpy': rpy_offset})
        ET.SubElement(fixed_joint, 'parent', attrib={'link': parent_link})
        ET.SubElement(fixed_joint, 'child', attrib={'link': prefix + hand_root_link})
        
        root_base.append(fixed_joint)

    # ==========================================
    # RPY OFFSETS
    # Because the left hand is physically mirrored now, 
    # you may need to invert one of the axis signs here!
    # ==========================================
    right_hand_rpy = "1.5708 0 1.5708" 
    left_hand_rpy = "-1.5708 0 1.5708" 

    # Attach the right hand
    append_hand(
        right_hand_urdf_path, 
        'right_hand', 
        'mano_right_', 
        'human_to_right_mano',
        rpy_offset=right_hand_rpy
    )
    
    # Attach the left hand
    append_hand(
        left_hand_urdf_path, 
        'left_hand', 
        'mano_left_', 
        'human_to_left_mano',
        rpy_offset=left_hand_rpy
    )
    
    # Save the unified URDF
    tree_base.write(output_urdf, encoding='utf-8', xml_declaration=True)
    print(f"\n[SUCCESS] Unified URDF saved to: {output_urdf}")

if __name__ == "__main__":
    merge_urdfs(
        base_urdf='model/urdf/human.urdf', 
        right_hand_urdf_path='/Users/theophile/Downloads/comfi-examples_new/model/mano-urdf/urdf/mano.urdf', 
        left_hand_urdf_path='/Users/theophile/Downloads/comfi-examples_new/model/mano-urdf/urdf/mano_left.urdf',
        output_urdf='/Users/theophile/Downloads/comfi-examples_new/model/urdf/human_with_hands.urdf'
    )