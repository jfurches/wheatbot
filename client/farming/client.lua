agent = require('agent')

addr, port = '127.0.0.1', 8080
endpoint = 'http://' .. addr .. ':' .. tostring(port) .. '/'

local function getAction(req_data)
    req = http.post(endpoint, textutils.serialiseJSON(req_data))

    if req == nil then
        print('Failed to send post request')
        return nil
    end

    data = req.readAll()
    data = textutils.unserializeJSON(data)

    if data and data.body then
        return data.body.action
    else
        return nil
    end
end

timesteps = 240
chestLoc = vector.new(-46, 66, -411)
fieldLoc = vector.new(-37, 66, -412)

robot = agent.new(240, chestLoc, fieldLoc)

while true do
    print('New episode')
    obs, info = robot:reset()
    done, truncated = false, false
    req_data = {
        type = 'reset',
        obs = obs,
        info = info
    }
    
    action = getAction(req_data)
    while not done and not truncated do
        if action == nil then
            print('Null action received')
            while true do
            end
        end

        obs, reward, done, truncated, info = robot:step(action)
        req_data = {
            type = 'step',
            obs = obs,
            reward = reward,
            done = done,
            truncated = truncated,
            info = info
        }
        action = getAction(req_data)
    end

    print('Finished episode')
end