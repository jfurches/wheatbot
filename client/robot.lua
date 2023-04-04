local robot = {
    actions = {"left", "right", "forward", "up", "down", "mine", "mineUp", "mineDown", "refuel"}
}

-- Gets the valid actions the robot can perform
function robot.getActions()
    local v1 = {}

    for k, action in pairs(robot.actions) do
        if robot.canDo(action) then
            v1[#v1+1] = action
        end
    end

    return v1
end

-- Returns the cost of performing an action
function robot.getCost(action)
    -- Movement has a cost of 1, left and right are turning
    if table.contains({"up", "down", "forward", "back"}, action) then
        return 1
    else
        return 0
    end
end

-- Performs the action, and returns the reward
function robot.doAction(action)
    local success = true
    local reward = robot.getCost(action)

    if action == "left" then
        success = turtle.turnLeft()
    elseif action == "right" then
        success = turtle.turnRight()
    elseif action == "forward" then
        success = turtle.forward()
    elseif action == "up" then
        success = turtle.up()
    elseif action == "down" then
        success = turtle.down()
    elseif action == "mine" then
        success = turtle.dig()
    elseif action == "mineUp" then
        success = turtle.digUp()
    elseif action == "mineDown" then
        success = turtle.digDown()
    elseif action == "refuel" then
        local currentFuel = turtle.getFuelLevel()
        for i=1,16,1 do
            turtle.select(i)
            success = turtle.refuel(1)
            if success then
                reward = turtle.getFuelLevel() - currentFuel
                break
            end
        end
    end

    return (success and reward) or 0
end

-- Tells whether or not the robot can perform the desired action
function robot.canDo(action)
    local canMove = turtle.getFuelLevel() > 0

    if action == "left" or action == "right" then
        return true
    elseif action == "forward" and canMove and not turtle.detect() then
        return true
    elseif action == "up" and canMove and not turtle.detectUp() then
        return true
    elseif action == "down" and canMove and not turtle.detectDown() then
        return true
    elseif action == "mine" and turtle.detect() then
        return true
    elseif action == "mineUp" and turtle.detectUp() then
        return true
    elseif action == "mineDown" and turtle.detectDown() then
        return true
    elseif action == "refuel" then
        for i=1,16,1 do
            turtle.select(i)
            -- by setting to 0 we can check if it's fuel without consuming
            if turtle.refuel(0) then
                return true
            end
        end

        return false
    else
        return false
    end
end

function table.contains(t, el)
    for k,v in pairs(t) do
        if el == v then
            return true
        end
    end

    return false
end

return robot