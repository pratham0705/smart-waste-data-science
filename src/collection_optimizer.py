def assign_collection_priority(risk_score, hotspot_level):
    if risk_score >= 75 or hotspot_level == "Critical Hotspot":
        return "Priority 1 - Immediate Collection"
    elif risk_score >= 50 or hotspot_level == "High Hotspot":
        return "Priority 2 - Daily Collection"
    elif risk_score >= 25 or hotspot_level == "Medium Hotspot":
        return "Priority 3 - Alternate Day Collection"
    else:
        return "Priority 4 - Regular Collection"


def assign_vehicle_requirement(waste):
    if waste >= 30000:
        return "3 Large Collection Vehicles"
    elif waste >= 20000:
        return "2 Medium Collection Vehicles"
    elif waste >= 10000:
        return "1 Medium Collection Vehicle"
    else:
        return "1 Small Collection Vehicle"


def assign_action_plan(priority):
    if "Priority 1" in priority:
        return "Immediate collection, landfill monitoring, recycling drive, emergency response team."
    elif "Priority 2" in priority:
        return "Daily collection, improve segregation, monitor landfill capacity."
    elif "Priority 3" in priority:
        return "Alternate-day collection, awareness campaigns, preventive monitoring."
    else:
        return "Regular collection schedule and periodic monitoring."
    