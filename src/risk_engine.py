def calculate_risk(waste, recycling, efficiency, awareness, landfill):
    
    score = 0

    # Waste impact
    if waste > 120000:
        score += 30
    elif waste > 80000:
        score += 20
    else:
        score += 10

    # Recycling impact
    if recycling < 40:
        score += 20
    elif recycling < 60:
        score += 10

    # Efficiency impact
    if efficiency < 5:
        score += 15
    elif efficiency < 7:
        score += 8

    # Awareness impact
    if awareness < 10:
        score += 10

    # Landfill capacity impact
    if landfill < 40000:
        score += 25
    elif landfill < 70000:
        score += 15

    return min(score, 100)


def risk_level(score):
    if score >= 75:
        return "Critical 🔴"
    elif score >= 50:
        return "High 🟠"
    elif score >= 25:
        return "Medium 🟡"
    else:
        return "Low 🟢"
    