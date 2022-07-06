def get_dynamic_target_attack_goal(model_preds, target_value, target_ranges):
    """ 
    Obtain dynamic target attack goal list used for experiments
    for both positive target and negative target 

    Args:
        model_preds: 
            A list contains the original model prediction for all windows
        target_value: 
            A float specifies the attack target, should be range within [0,1]
            since the training time series is normalized within [0,1]
        target_ranges: 
            A list contains the range of the target for all windows

    Returns:
        Two lists contain the positive and negative target attack goal
    """
    positive_attack_goals, negative_attack_goals = [], []

    for window_index in range(model_preds.shape[0]):
        # For each window, generate the attack target based on: original model prediction +/- target value times target range
        positive_attack_goals.append((window_index, model_preds[window_index].item() + target_value * target_ranges[window_index]))
        negative_attack_goals.append((window_index, model_preds[window_index].item() - target_value * target_ranges[window_index]))

    return positive_attack_goals, negative_attack_goals