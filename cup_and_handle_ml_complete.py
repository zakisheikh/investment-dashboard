def detect_cup_and_handle_in_window(window):
    close_prices = window['Close'].values
    volume = window['Volume'].values

    # Parameters for cup and handle detection
    min_cup_length = 10
    max_cup_length = 60
    min_handle_length = 5
    max_handle_length = 20
    min_depth = 0.1  # Minimum depth of the cup (10%)
    max_depth = 0.5  # Maximum depth of the cup (50%)
    handle_max_retrace = 0.5  # Handle retracement should not exceed 50% of cup depth

    # Step 1: Identify the cup
    cup_found = False
    for i in range(min_cup_length, min(max_cup_length, len(close_prices) - min_handle_length)):
        left_peak = close_prices[0]
        right_peak = close_prices[i]
        cup_bottom = np.min(close_prices[:i+1])
        bottom_index = np.argmin(close_prices[:i+1])

        # Ensure the bottom is not at the edges
        if bottom_index == 0 or bottom_index == i:
            continue

        # Calculate cup depth
        peak_average = (left_peak + right_peak) / 2
        depth = (peak_average - cup_bottom) / peak_average

        if depth < min_depth or depth > max_depth:
            continue

        # Step 2: Identify the handle
        for j in range(i + min_handle_length, min(i + max_handle_length, len(close_prices))):
            handle = close_prices[i+1:j+1]
            handle_max = np.max(handle)
            handle_min = np.min(handle)

            # Handle should not exceed the peaks
            if handle_max > peak_average:
                continue

            # Handle retracement should not exceed 50% of cup depth
            handle_retrace = (handle_max - handle_min) / (peak_average - cup_bottom)
            if handle_retrace > handle_max_retrace:
                continue

            # If all conditions are met, pattern is found
            cup_found = True
            break

        if cup_found:
            break

    label = 1 if cup_found else 0
    return label
