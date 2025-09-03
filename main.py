import cv2
import numpy as np
import math

img_1_stars_red = []
img_1_stars_yellow = []
img_1_stars_green = []
img_2_stars_red = []
img_2_stars_yellow = []
img_2_stars_green = []
img_3_stars_red = []
img_3_stars_yellow = []
img_3_stars_green = []
img_4_stars_red = []
img_4_stars_yellow = []
img_4_stars_green = []
img_5_stars_red = []
img_5_stars_yellow = []
img_5_stars_green = []
img_6_stars_red = []
img_6_stars_yellow = []
img_6_stars_green = []
img_7_stars_red = []
img_7_stars_yellow = []
img_7_stars_green = []
img_8_stars_red = []
img_8_stars_yellow = []
img_8_stars_green = []
img_9_stars_red = []
img_9_stars_yellow = []
img_9_stars_green = []
img_10_stars_red = []
img_10_stars_yellow = []
img_10_stars_green = []




img_1_squares_red = []
img_1_squares_yellow = []
img_1_squares_green = []
img_2_squares_red = []
img_2_squares_yellow = []
img_2_squares_green = []
img_3_squares_red = []
img_3_squares_yellow = []
img_3_squares_green = []
img_4_squares_red = []
img_4_squares_yellow = []
img_4_squares_green = []
img_5_squares_red = []
img_5_squares_yellow = []
img_5_squares_green = []
img_6_squares_red = []
img_6_squares_yellow = []
img_6_squares_green = []
img_7_squares_red = []
img_7_squares_yellow = []
img_7_squares_green = []
img_8_squares_red = []
img_8_squares_yellow = []
img_8_squares_green = []
img_9_squares_red = []
img_9_squares_yellow = []
img_9_squares_green = []
img_10_squares_red = []
img_10_squares_yellow = []
img_10_squares_green = []





img_1_triangles_red = []
img_1_triangles_yellow = []
img_1_triangles_green = []
img_2_triangles_red = []
img_2_triangles_yellow = []
img_2_triangles_green = []
img_3_triangles_red = []
img_3_triangles_yellow = []
img_3_triangles_green = []
img_4_triangles_red = []
img_4_triangles_yellow = []
img_4_triangles_green = []
img_5_triangles_red = []
img_5_triangles_yellow = []
img_5_triangles_green = []
img_6_triangles_red = []
img_6_triangles_yellow = []
img_6_triangles_green = []
img_7_triangles_red = []
img_7_triangles_yellow = []
img_7_triangles_green = []
img_8_triangles_red = []
img_8_triangles_yellow = []
img_8_triangles_green = []
img_9_triangles_red = []
img_9_triangles_yellow = []
img_9_triangles_green = []
img_10_triangles_red = []
img_10_triangles_yellow = []
img_10_triangles_green = []


pink_pads = []
grey_pads = []
blue_pads = []



def get_color_name(hsv_pixel):
 
    h, s, v = hsv_pixel

 
    if s < 25 and v > 150:
        return "Grey"

    elif h > 135 and h < 155:
        return "Pink"
   
    elif h > 95 and h < 125:
        return "Blue"
 
    elif h > 35 and h < 75: 
        return "Green"
   
    elif h > 20 and h < 30:
        return "Yellow"
  
    elif h < 15 or h > 170:
        return "Red"
    else:
        return "Unknown"


counter = [1,2,3,4,5,6,7,8,9,10]
count = 0

images = []
for i in range(1, 11):
    file_name = f"task_images\\{i}.png"
    image = cv2.imread(file_name)
    if image is not None:
        images.append(image)





recolored_images = []
count = -1
for i in images:
    count += 1
    img = images[count]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 180])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    new_color = np.array([255, 255, 255], dtype=np.uint8)  
    flat = img.copy()
    flat[mask > 0] = new_color
    recolored_images.append(flat)
    cv2.imshow("Differentiated Ocean and Land", flat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



count = 0
for idx, img in enumerate(images):
    count += 1
    output = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  
    lower_green_bg = np.array([35, 40, 40])
    upper_green_bg = np.array([85, 255, 180]) 

   
    lower_blue_bg = np.array([90, 70, 40])
    upper_blue_bg = np.array([125, 255, 180])
    
   
    mask_green_bg = cv2.inRange(hsv, lower_green_bg, upper_green_bg)
    mask_blue_bg = cv2.inRange(hsv, lower_blue_bg, upper_blue_bg)
    
   
    background_mask = cv2.bitwise_or(mask_green_bg, mask_blue_bg)
    
    
    shapes_mask = cv2.bitwise_not(background_mask)
    
 
    kernel = np.ones((5,5), np.uint8)
    shapes_mask = cv2.morphologyEx(shapes_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

  
    contours, _ = cv2.findContours(shapes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100: 
            continue

       
        shape = "Unknown"
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
        num_vertices = len(approx)

        if num_vertices == 3:
            shape = "Triangle"
        elif num_vertices == 4:
            shape = "Square"
        else:
            solidity = float(area) / cv2.contourArea(cv2.convexHull(cnt))
            if solidity > 0.9:
                shape = "Circle"
            elif solidity < 0.7: 
                shape = "Star"

        if shape != "Unknown":
          
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                hsv_pixel = hsv[cy, cx]
                color_name = get_color_name(hsv_pixel)
                
                
                if shape == "Star":
                    if count == 1:
                        if color_name == "Red":
                            img_1_stars_red.append([cx,cy,9])
                        elif color_name == "Yellow":
                            img_1_stars_yellow.append([cx,cy,6])
                        elif color_name == "Green":
                            img_1_stars_green.append([cx,cy,3])
                    elif count == 2:
                        if color_name == "Red":
                            img_2_stars_red.append([cx,cy,9])
                        elif color_name == "Yellow":
                            img_2_stars_yellow.append([cx,cy,6])
                        elif color_name == "Green":
                            img_2_stars_green.append([cx,cy,3])
                    elif count == 3:
                        if color_name == "Red":
                            img_3_stars_red.append([cx,cy,9])
                        elif color_name == "Yellow":
                            img_3_stars_yellow.append([cx,cy,6])
                        elif color_name == "Green":
                            img_3_stars_green.append([cx,cy,3])
                    elif count == 4:
                        if color_name == "Red":
                            img_4_stars_red.append([cx,cy,9])
                        elif color_name == "Yellow":
                            img_4_stars_yellow.append([cx,cy,6])
                        elif color_name == "Green":
                            img_4_stars_green.append([cx,cy,3])
                    elif count == 5:
                        if color_name == "Red":
                            img_5_stars_red.append([cx,cy,9])
                        elif color_name == "Yellow":
                            img_5_stars_yellow.append([cx,cy,6])
                        elif color_name == "Green":
                            img_5_stars_green.append([cx,cy,3])
                    elif count == 6:
                        if color_name == "Red":
                            img_6_stars_red.append([cx,cy,9])
                        elif color_name == "Yellow":
                            img_6_stars_yellow.append([cx,cy,6])
                        elif color_name == "Green":
                            img_6_stars_green.append([cx,cy,3])
                    elif count == 7:
                        if color_name == "Red":
                            img_7_stars_red.append([cx,cy,9])
                        elif color_name == "Yellow":
                            img_7_stars_yellow.append([cx,cy,6])
                        elif color_name == "Green":
                            img_7_stars_green.append([cx,cy,3])
                    elif count == 8:
                        if color_name == "Red":
                            img_8_stars_red.append([cx,cy,9])
                        elif color_name == "Yellow":
                            img_8_stars_yellow.append([cx,cy,6])
                        elif color_name == "Green":
                            img_8_stars_green.append([cx,cy,3])
                    elif count == 9:
                        if color_name == "Red":
                            img_9_stars_red.append([cx,cy,9])
                        elif color_name == "Yellow":
                            img_9_stars_yellow.append([cx,cy,6])
                        elif color_name == "Green":
                            img_9_stars_green.append([cx,cy,3])
                    elif count == 10:
                        if color_name == "Red":
                            img_10_stars_red.append([cx,cy,9])
                        elif color_name == "Yellow":
                            img_10_stars_yellow.append([cx,cy,6])
                        elif color_name == "Green":
                            img_10_stars_green.append([cx,cy,3])

                elif shape == "Square":
                    if count == 1:
                        if color_name == "Red":
                            img_1_squares_red.append([cx,cy,3])
                        elif color_name == "Yellow":
                            img_1_squares_yellow.append([cx,cy,2])
                        elif color_name == "Green":
                            img_1_squares_green.append([cx,cy,1]) 
                    elif count == 2:
                        if color_name == "Red":
                            img_2_squares_red.append([cx,cy,3])
                        elif color_name == "Yellow":
                            img_2_squares_yellow.append([cx,cy,2])
                        elif color_name == "Green":
                            img_2_squares_green.append([cx,cy,1])
                    elif count == 3:
                        if color_name == "Red":
                            img_3_squares_red.append([cx,cy,3])
                        elif color_name == "Yellow":
                            img_3_squares_yellow.append([cx,cy,2])
                        elif color_name == "Green":
                            img_3_squares_green.append([cx,cy,1])
                    elif count == 4:
                        if color_name == "Red":
                            img_4_squares_red.append([cx,cy,3])
                        elif color_name == "Yellow":
                            img_4_squares_yellow.append([cx,cy,2])
                        elif color_name == "Green":
                            img_4_squares_green.append([cx,cy,1])
                    elif count == 5:
                        if color_name == "Red":
                            img_5_squares_red.append([cx,cy,3])
                        elif color_name == "Yellow":
                            img_5_squares_yellow.append([cx,cy,2])
                        elif color_name == "Green":
                            img_5_squares_green.append([cx,cy,1])
                    elif count == 6:
                        if color_name == "Red":
                            img_6_squares_red.append([cx,cy,3])
                        elif color_name == "Yellow":
                            img_6_squares_yellow.append([cx,cy,2])
                        elif color_name == "Green":
                            img_6_squares_green.append([cx,cy,1])
                    elif count == 7:
                        if color_name == "Red":
                            img_7_squares_red.append([cx,cy,3])
                        elif color_name == "Yellow":
                            img_7_squares_yellow.append([cx,cy,2])
                        elif color_name == "Green":
                            img_7_squares_green.append([cx,cy,1])
                    elif count == 8:
                        if color_name == "Red":
                            img_8_squares_red.append([cx,cy,3])
                        elif color_name == "Yellow":
                            img_8_squares_yellow.append([cx,cy,2])
                        elif color_name == "Green":
                            img_8_squares_green.append([cx,cy,1])
                    elif count == 9:
                        if color_name == "Red":
                            img_9_squares_red.append([cx,cy,3])
                        elif color_name == "Yellow":
                            img_9_squares_yellow.append([cx,cy,2])
                        elif color_name == "Green":
                            img_9_squares_green.append([cx,cy,1])
                    elif count == 10:
                        if color_name == "Red":
                            img_10_squares_red.append([cx,cy,3])
                        elif color_name == "Yellow":
                            img_10_squares_yellow.append([cx,cy,2])
                        elif color_name == "Green":
                            img_10_squares_green.append([cx,cy,1])
                elif shape == "Triangle":
                    if count == 1:
                        if color_name == "Red":
                            img_1_triangles_red.append([cx,cy,6])
                        elif color_name == "Yellow":
                            img_1_triangles_yellow.append([cx,cy,4])
                        elif color_name == "Green":
                            img_1_triangles_green.append([cx,cy,2])
                    elif count == 2:
                        if color_name == "Red":
                            img_2_triangles_red.append([cx,cy,6])
                        elif color_name == "Yellow":
                            img_2_triangles_yellow.append([cx,cy,4])
                        elif color_name == "Green":
                            img_2_triangles_green.append([cx,cy,2])
                    elif count == 3:
                        if color_name == "Red":
                            img_3_triangles_red.append([cx,cy,6])
                        elif color_name == "Yellow":
                            img_3_triangles_yellow.append([cx,cy,4])
                        elif color_name == "Green":
                            img_3_triangles_green.append([cx,cy,2])
                    elif count == 4:
                        if color_name == "Red":
                            img_4_triangles_red.append([cx,cy,6])
                        elif color_name == "Yellow":
                            img_4_triangles_yellow.append([cx,cy,4])
                        elif color_name == "Green":
                            img_4_triangles_green.append([cx,cy,2])
                    elif count == 5:
                        if color_name == "Red":
                            img_5_triangles_red.append([cx,cy,6])
                        elif color_name == "Yellow":
                            img_5_triangles_yellow.append([cx,cy,4])
                        elif color_name == "Green":
                            img_5_triangles_green.append([cx,cy,2])
                    elif count == 6:
                        if color_name == "Red":
                            img_6_triangles_red.append([cx,cy,6])
                        elif color_name == "Yellow":
                            img_6_triangles_yellow.append([cx,cy,4])
                        elif color_name == "Green":
                            img_6_triangles_green.append([cx,cy,2])
                    elif count == 7:
                        if color_name == "Red":
                            img_7_triangles_red.append([cx,cy,6])
                        elif color_name == "Yellow":
                            img_7_triangles_yellow.append([cx,cy,4])
                        elif color_name == "Green":
                            img_7_triangles_green.append([cx,cy,2])
                    elif count == 8:
                        if color_name == "Red":
                            img_8_triangles_red.append([cx,cy,6])
                        elif color_name == "Yellow":
                            img_8_triangles_yellow.append([cx,cy,4])
                        elif color_name == "Green":
                            img_8_triangles_green.append([cx,cy,2])
                    elif count == 9:
                        if color_name == "Red":
                            img_9_triangles_red.append([cx,cy,6])
                        elif color_name == "Yellow":
                            img_9_triangles_yellow.append([cx,cy,4])
                        elif color_name == "Green":
                            img_9_triangles_green.append([cx,cy,2])
                    elif count == 10:
                        if color_name == "Red":
                            img_10_triangles_red.append([cx,cy,6])
                        elif color_name == "Yellow":
                            img_10_triangles_yellow.append([cx,cy,4])
                        elif color_name == "Green":
                            img_10_triangles_green.append([cx,cy,2])
                elif shape == "Circle":
                    if color_name == "Pink":
                        pink_pads.append([cx,cy])
                    elif color_name == "Grey":
                        grey_pads.append([cx,cy])
                    elif color_name == "Blue":
                        blue_pads.append([cx,cy])
                else:
                    None

                x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(approx.reshape(-1, 2))
                
                label = f"{color_name} {shape}"
                cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)
                cv2.putText(output, label, (x_bbox-20, y_bbox-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    
    cv2.imshow("Detected Casualties", output)
    cv2.waitKey(0)
cv2.destroyAllWindows()




all_images_casualties = []
for i in range(1, 11):
    casualties = []
    # Gather all casualties for the current image 'i'
    for casualty_list in [
        globals().get(f'img_{i}_stars_red', []), globals().get(f'img_{i}_stars_yellow', []), globals().get(f'img_{i}_stars_green', []),
        globals().get(f'img_{i}_squares_red', []), globals().get(f'img_{i}_squares_yellow', []), globals().get(f'img_{i}_squares_green', []),
        globals().get(f'img_{i}_triangles_red', []), globals().get(f'img_{i}_triangles_yellow', []), globals().get(f'img_{i}_triangles_green', [])
    ]:
        for x, y, priority in casualty_list:
            casualties.append({'coords': (x, y), 'priority': priority})
    all_images_casualties.append(casualties)




def calculate_distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)





final_results = []
for i in range(10):  # Corresponds to images 1-10

    
    # Sort casualties by priority score in descending order
    sorted_casualties = sorted(all_images_casualties[i], key=lambda c: c['priority'], reverse=True)
    
    # Initialize camps for this image
    camps = {
        'blue': {'coords': tuple(blue_pads[i]), 'capacity': 4, 'assignments': [], 'total_priority': 0},
        'pink': {'coords': tuple(pink_pads[i]), 'capacity': 3, 'assignments': [], 'total_priority': 0},
        'grey': {'coords': tuple(grey_pads[i]), 'capacity': 2, 'assignments': [], 'total_priority': 0}
    }
    
    # Assignment process
    for casualty in sorted_casualties:
        distances_to_camps = []
        for name, data in camps.items():
            dist = calculate_distance(casualty['coords'], data['coords'])
            distances_to_camps.append({'name': name, 'dist': dist})
            
        # Sort camps by distance (closest first)
        sorted_camps_by_dist = sorted(distances_to_camps, key=lambda c: c['dist'])
        
        # Find the closest available camp and assign the casualty
        for camp_option in sorted_camps_by_dist:
            camp_name = camp_option['name']
            if camps[camp_name]['capacity'] > 0:
                camps[camp_name]['assignments'].append(casualty)
                camps[camp_name]['total_priority'] += casualty['priority']
                camps[camp_name]['capacity'] -= 1
                break  # Move to the next casualty once assigned

    # Calculate final stats for this image
    total_assigned = sum(len(d['assignments']) for d in camps.values())
    total_priority = sum(d['total_priority'] for d in camps.values())
    pr = total_priority / total_assigned if total_assigned > 0 else 0
    
    final_results.append({
        'image_name': f"Image {i+1}",
        'camps': camps,
        'pr': pr
    })

# PRINT RESULTS 


print("FINAL RESULTS")


# Counts and Details
print("Casualty Assignment and Details")
for result in final_results:
    print(f"--- {result['image_name']} ---")
    print(f"a) Casualty Counts:")
    for name, data in result['camps'].items():
        print(f"   - {name.capitalize()} Camp: {len(data['assignments'])}")
    print(f"b) Details:")
    for name, data in result['camps'].items():
        priorities = [c['priority'] for c in data['assignments']]
        print(f"   - {name.capitalize()} assigned priorities: {priorities}")
    print("-" * 20)

# Priorities and Rescue Ratio
print("Total Priorities and Rescue Ratios")
for result in final_results:
    priorities = [result['camps']['blue']['total_priority'], result['camps']['pink']['total_priority'], result['camps']['grey']['total_priority']]
    print(f"- **{result['image_name']}**")
    print(f"  - Total Priority of Camps [Blue, Pink, Grey]: {priorities}")
    print(f"  - Average Rescue Ratio (Pr): {result['pr']:.2f}\n")

# Image Ranking
print("Image Ranking by Rescue Ratio (Pr)")
sorted_by_pr = sorted(final_results, key=lambda x: x['pr'], reverse=True)
for i, result in enumerate(sorted_by_pr):
    print(f"{i+1}. {result['image_name']} (Pr: {result['pr']:.2f})")






# DISPLAYING THE ROUTES AS PER THE OUTPUT ACHEIVED

img1 = recolored_images[0]
img2 = recolored_images[1]
img3 = recolored_images[2]
img4 = recolored_images[3]
img5 = recolored_images[4]
img6 = recolored_images[5]
img7 = recolored_images[6]
img8 = recolored_images[7]
img9 = recolored_images[8]
img10 = recolored_images[9]

casualty_routes = []

# IMAGE 1 SOLUTIONS
image1 = cv2.line(img1, blue_pads[0], (img_1_stars_red[0][:2]), (0,0,0), 2)
image1 = cv2.line(img1, blue_pads[0], (img_1_squares_red[0][:2]), (0,0,0), 2)
image1 = cv2.line(img1, pink_pads[0], (img_1_stars_green[0][:2]), (0,0,0), 2)
image1 = cv2.line(img1, pink_pads[0], (img_1_triangles_yellow[1][:2]), (0,0,0), 2)
image1 = cv2.line(img1, grey_pads[0], (img_1_triangles_yellow[0][:2]), (0,0,0), 2)
image1 = cv2.line(img1, grey_pads[0], (img_1_stars_yellow[0][:2]), (0,0,0), 2)
casualty_routes.append(image1)

# IMAGE 2 SOLUTIONS
image2 = cv2.line(img2, blue_pads[1], (img_2_stars_red[0][:2]), (0,0,0), 2)
image2 = cv2.line(img2, blue_pads[1], (img_2_triangles_yellow[0][:2]), (0,0,0), 2)
image2 = cv2.line(img2, blue_pads[1], (img_2_stars_green[0][:2]), (0,0,0), 2)
image2 = cv2.line(img2, pink_pads[1], (img_2_squares_yellow[0][:2]), (0,0,0), 2)
image2 = cv2.line(img2, pink_pads[1], (img_2_triangles_red[0][:2]), (0,0,0), 2)
image2 = cv2.line(img2, pink_pads[1], (img_2_squares_green[0][:2]), (0,0,0), 2)
image2 = cv2.line(img2, grey_pads[1], (img_2_squares_red[0][:2]), (0,0,0), 2)
image2 = cv2.line(img2, grey_pads[1], (img_2_stars_yellow[0][:2]), (0,0,0), 2)
casualty_routes.append(image2)


# IMAGE 3 SOLUTIONS
image3 = cv2.line(img3, blue_pads[2], (img_3_squares_green[0][:2]), (0,0,0), 2)
image3 = cv2.line(img3, blue_pads[2], (img_3_squares_green[1][:2]), (0,0,0), 2)
image3 = cv2.line(img3, blue_pads[2], (img_3_squares_yellow[0][:2]), (0,0,0), 2)
image3 = cv2.line(img3, blue_pads[2], (img_3_squares_red[0][:2]), (0,0,0), 2)
image3 = cv2.line(img3, pink_pads[2], (img_3_stars_yellow[0][:2]), (0,0,0), 2)
image3 = cv2.line(img3, pink_pads[2], (img_3_triangles_yellow[1][:2]), (0,0,0), 2)
image3 = cv2.line(img3, pink_pads[2], (img_3_stars_green[0][:2]), (0,0,0), 2)
image3 = cv2.line(img3, grey_pads[2], (img_3_stars_red[0][:2]), (0,0,0), 2)
image3 = cv2.line(img3, grey_pads[2], (img_3_triangles_yellow[0][:2]), (0,0,0), 2)
casualty_routes.append(image3)

# IMAGE 4 SOLUTIONS
image4 = cv2.line(img4, blue_pads[3], (img_4_squares_green[0][:2]), (0,0,0), 2)
image4 = cv2.line(img4, blue_pads[3], (img_4_squares_green[1][:2]), (0,0,0), 2)
image4 = cv2.line(img4, blue_pads[3], (img_4_squares_yellow[0][:2]), (0,0,0), 2)
image4 = cv2.line(img4, blue_pads[3], (img_4_squares_red[0][:2]), (0,0,0), 2)
image4 = cv2.line(img4, pink_pads[3], (img_4_stars_red[0][:2]), (0,0,0), 2)
image4 = cv2.line(img4, pink_pads[3], (img_4_triangles_yellow[0][:2]), (0,0,0), 2)
image4 = cv2.line(img4, pink_pads[3], (img_4_triangles_yellow[1][:2]), (0,0,0), 2)
image4 = cv2.line(img4, grey_pads[3], (img_4_stars_red[1][:2]), (0,0,0), 2)
image4 = cv2.line(img4, grey_pads[3], (img_4_triangles_red[0][:2]), (0,0,0), 2)
casualty_routes.append(image4)

# IMAGE 5 SOLUTIONS
image5 = cv2.line(img5, blue_pads[4], (img_5_triangles_red[0][:2]), (0,0,0), 2)

image5 = cv2.line(img5, pink_pads[4], (img_5_stars_red[0][:2]), (0,0,0), 2)
image5 = cv2.line(img5, pink_pads[4], (img_5_stars_green[0][:2]), (0,0,0), 2)

image5 = cv2.line(img5, grey_pads[4], (img_5_squares_red[0][:2]), (0,0,0), 2)
image5 = cv2.line(img5, grey_pads[4], (img_5_triangles_yellow[0][:2]), (0,0,0), 2)
casualty_routes.append(image5)

# IMAGE 6 SOLUTIONS
image6 = cv2.line(img6, blue_pads[5], (img_6_squares_red[0][:2]), (0,0,0), 2)
image6 = cv2.line(img6, blue_pads[5], (img_6_squares_red[1][:2]), (0,0,0), 2)
image6 = cv2.line(img6, blue_pads[5], (img_6_stars_green[0][:2]), (0,0,0), 2)

image6 = cv2.line(img6, pink_pads[5], (img_6_stars_yellow[0][:2]), (0,0,0), 2)
image6 = cv2.line(img6, pink_pads[5], (img_6_triangles_red[0][:2]), (0,0,0), 2)

image6 = cv2.line(img6, grey_pads[5], (img_6_triangles_green[0][:2]), (0,0,0), 2)
image6 = cv2.line(img6, grey_pads[5], (img_6_stars_red[0][:2]), (0,0,0), 2)
casualty_routes.append(image6)

# IMAGE 7 SOLUTIONS
image7 = cv2.line(img7, blue_pads[6], (img_7_squares_red[0][:2]), (0,0,0), 2)
image7 = cv2.line(img7, blue_pads[6], (img_7_squares_yellow[0][:2]), (0,0,0), 2)
image7 = cv2.line(img7, blue_pads[6], (img_7_stars_green[0][:2]), (0,255,0), 2)

image7 = cv2.line(img7, pink_pads[6], (img_7_stars_red[0][:2]), (0,0,0), 2)
image7 = cv2.line(img7, pink_pads[6], (img_7_triangles_yellow[0][:2]), (0,0,0), 2)
image7 = cv2.line(img7, pink_pads[6], (img_7_triangles_green[0][:2]), (0,0,0), 2)

image7 = cv2.line(img7, grey_pads[6], (img_7_stars_red[1][:2]), (0,0,0), 2)
image7 = cv2.line(img7, grey_pads[6], (img_7_triangles_red[0][:2]), (0,0,0), 2)
casualty_routes.append(image7)

# IMAGE 8 SOLUTIONS
image8 = cv2.line(img8, blue_pads[7], (img_8_squares_red[0][:2]), (0,0,0), 2)
image8 = cv2.line(img8, blue_pads[7], (img_8_squares_yellow[1][:2]), (0,0,0), 2)
image8 = cv2.line(img8, blue_pads[7], (img_8_triangles_green[0][:2]), (0,0,0), 2)
image8 = cv2.line(img8, blue_pads[7], (img_8_triangles_yellow[0][:2]), (0,0,0), 2)

image8 = cv2.line(img8, pink_pads[7], (img_8_stars_red[0][:2]), (0,0,0), 2)
image8 = cv2.line(img8, pink_pads[7], (img_8_squares_yellow[0][:2]), (0,0,0), 2)
image8 = cv2.line(img8, pink_pads[7], (img_8_stars_green[0][:2]), (0,0,0), 2)

image8 = cv2.line(img8, grey_pads[7], (img_8_stars_red[1][:2]), (0,0,0), 2)
image8 = cv2.line(img8, grey_pads[7], (img_8_triangles_red[0][:2]), (0,0,0), 2)
casualty_routes.append(image8)

# IMAGE 9 SOLUTIONS
image9 = cv2.line(img9, blue_pads[8], (img_9_squares_red[0][:2]), (0,0,0), 2)
image9 = cv2.line(img9, blue_pads[8], (img_9_squares_yellow[0][:2]), (0,0,0), 2)
image9 = cv2.line(img9, blue_pads[8], (img_9_squares_green[1][:2]), (0,0,0), 2)


image9 = cv2.line(img9, pink_pads[8], (img_9_stars_red[0][:2]), (0,0,0), 2)
image9 = cv2.line(img9, pink_pads[8], (img_9_squares_green[0][:2]), (0,0,0), 2)

image9 = cv2.line(img9, grey_pads[8], (img_9_stars_green[0][:2]), (0,0,0), 2)
image9 = cv2.line(img9, grey_pads[8], (img_9_stars_yellow[0][:2]), (0,0,0), 2)
casualty_routes.append(image9)


# IMAGE 10 SOLUTIONS
image10 = cv2.line(img10, blue_pads[9], (img_10_triangles_green[0][:2]), (0,0,0), 2)
image10 = cv2.line(img10, blue_pads[9], (img_10_triangles_green[2][:2]), (0,0,0), 2)
image10 = cv2.line(img10, blue_pads[9], (img_10_triangles_green[3][:2]), (0,0,0), 2)
image10 = cv2.line(img10, blue_pads[9], (img_10_squares_red[1][:2]), (0,0,0), 2)


image10 = cv2.line(img10, pink_pads[9], (img_10_stars_red[0][:2]), (0,0,0), 2)
image10 = cv2.line(img10, pink_pads[9], (img_10_triangles_green[1][:2]), (0,0,0), 2)
image10 = cv2.line(img10, pink_pads[9], (img_10_squares_red[0][:2]), (0,0,0), 2)


image10 = cv2.line(img10, grey_pads[9], (img_10_stars_yellow[0][:2]), (0,0,0), 2)
image10 = cv2.line(img10, grey_pads[9], (img_10_stars_yellow[1][:2]), (0,0,0), 2)


casualty_routes.append(image10)





for i in range(0,10):
    cv2.imshow("Routes", casualty_routes[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()