def run_agent(user, message):
    """
    Simple agent logic that echoes the message back with the user's ID.
    
    Args:
        user (dict): The user payload from the verified JWT.
        message (str): The message sent by the user.
        
    Returns:
        str: A response message.
    """
    user_id = user.get("sub", "Unknown User")
    return f"Agent received '{message}' from user {user_id}"
