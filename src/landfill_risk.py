def landfill_risk_score(waste, capacity, growth_rate):
    usage_ratio = waste / capacity if capacity > 0 else 1

    score = 0

    if usage_ratio > 0.9:
        score += 40
    elif usage_ratio > 0.7:
        score += 25

    if growth_rate > 5:
        score += 30
    elif growth_rate > 2:
        score += 15

    if capacity < 30000:
        score += 30

    return min(score, 100)


def landfill_status(score):
    if score >= 70:
        return "Critical 🔴"
    elif score >= 40:
        return "High 🟠"
    else:
        return "Moderate 🟡"