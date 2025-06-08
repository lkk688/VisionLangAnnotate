

prompts = {
    "vehicle": (
        "Analyze the vehicle in the image. "
        "Is it abandoned, burned, blocking a bike lane, fire hydrant, or red curb? "
        "Is it missing parts like tires or windows, or up on jacks?"
    ),
    "road_surface": (
        "Look at the road area. Do you see potholes, flooding or sewage backup in the bike lane, "
        "faded bike lane paint, or downed objects like bollards or power lines?"
    ),
    "road_sign": (
        "Does the sign appear broken, missing, obscured, or damaged in any way?"
    ),
    "traffic_light": (
        "Is the traffic light functioning properly? Is it broken or turned off?"
    ),
    "trash": (
        "Does the image show dumped trash, glass, yard waste, or debris?"
    ),
    "trash_can": (
        "Is a residential or commercial trash bin blocking the bike lane?"
    ),
    "tree": (
        "Is there a tree branch overhanging into the road or sidewalk?"
    ),
    "person": (
        "Is the person shown setting up a vendor cart or otherwise obstructing a bike lane or road?"
    ),
    "other": (
        "Does the image contain graffiti, construction signs or cones, or blocked roads?"
    )
}

#Scene-Aware and Category-Specific Prompts)
prompts_scene = {
    "vehicle": (
        "This object is a vehicle. Based on its placement, determine whether it is parked illegally, "
        "such as blocking a bike lane, fire hydrant, or red curb. "
        "Also assess if the vehicle appears to be abandoned, damaged, burned, on jacks, or missing parts like windows or tires."
    ),

    "road_surface": (
        "This is part of the road surface. Check for any road damage such as potholes, standing water or flooding in the bike lane, "
        "faded or unclear road or bike lane markings, or any fallen infrastructure like bollards or power lines."
    ),

    "road_sign": (
        "This object appears to be a road sign. Examine whether it is broken, obscured, missing, bent, or otherwise degraded."
    ),

    "traffic_light": (
        "This is a traffic signal. Determine whether it is functional, broken, misaligned, or not illuminated."
    ),

    "trash": (
        "This object appears to be a pile of trash or debris. Assess whether it is dumped illegally, includes yard waste, glass, or obstructs any public space."
    ),

    "trash_can": (
        "This is a trash container. Determine whether it is a residential bin or a commercial dumpster, and whether it is improperly placed—especially if it obstructs a bike lane or pedestrian path."
    ),

    "tree": (
        "This is a tree or branch. Assess if any part of it overhangs into the road, sidewalk, or bike lane in a way that could obstruct vehicles or street sweepers."
    ),

    "person": (
        "This is a person. Identify whether they are setting up a street vendor stand, particularly in a location that obstructs a bike lane or public path."
    ),

    "other": (
        "The highlighted area may include objects such as graffiti, construction signs, orange cones, or temporary blockades. "
        "Determine whether any of these are present and describe the nature of the obstruction or marking."
    )
}