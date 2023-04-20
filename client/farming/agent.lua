tracker = require('tracker')

function table.contains(t, el)
    for k,v in pairs(t) do
        if el == v then
            return true
        end
    end

    return false
end

local function oneHot(x, el)
    vec = {}
    for k, v in pairs(x) do
        if el == v then
            vec[k] = 1
        else
            vec[k] = 0
        end
    end
    
    return vec
end

local agent = {
    actions = {'no-op', 'move-forward', 'turn-left',
               'turn-right', 'harvest', 'interact-chest'},
    goals = {'goto-field', 'harvest-wheat', 'goto-chest'},
    blocks = {nil, 'air', 'wheat', 'water', 'dirt', 'grass',
              'farmland', 'chest', 'composter', 'fence'},

    reset = function(self)
        self.timesteps_remaining = self.max_timesteps
        self.tracker:setLocation()
        return self:getObs(), {}
    end,

    step = function(self, action)
        action = self.actions[action + 1] -- add +1 since python is 0 indexed
    
        reward = 0
        done = false
        truncated = false
        info = {
            wheat_collected = 0
        }
    
        self:fuelCheck()
    
        if action == 'no-op' then
            os.sleep(0.5)
        elseif action == 'move-forward' then
            self.tracker:forward() -- use tracker to move so we retain our position
        elseif action == 'turn-left' then
            self.tracker:turnLeft()
        elseif action == 'turn-right' then
            self.tracker:turnRight()
        elseif action == 'harvest' then
            block = self:getFacing()
            if block and block.name == 'wheat' then
                -- break wheat if mature, else leave it alone
                if block.age == 7 then
                    turtle.dig()
                    reward = reward + 0.01
    
                    -- we broke the wheat, so try to replant a seed
                    -- todo: future versions might include replant as
                    -- a separate action
                    if self:getFacing() == nil then
                        for i=1,16,1 do
                            item = turtle.getItemDetail(i)
                            if item and item.name == 'minecraft:seeds' then
                                oldSlot = turtle.getSelectedSlot()
                                turtle.select(i)
                                turtle.place()
                                turtle.select(oldSlot)
                                break
                            end
                        end
                    end
                else
                    return self:step(0) -- no-op for non-mature wheat
                end
            else
                return self:step(0) -- no-op for a non-wheat block
            end
        elseif action == 'interact-chest' then
            block = self:getFacing()
            if block and block.name == 'chest' then
                oldSlot = turtle.getSelectedSlot()
                for slot = 1,16,1 do
                    item = turtle.getItemDetail(slot)
                    if item and item.name == 'minecraft:wheat' then
                        turtle.select(slot)
                        amount = turtle.getItemCount(slot)
                        ret = turtle.drop()
                        if ret then
                            info.wheat_collected = info.wheat_collected + amount
                        end
                    end
                end
    
                turtle.select(oldSlot)
    
                -- Compute chest reward if we actually collected wheat
                if info.wheat_collected > 0 then
                    print('Collected ', info.wheat_collected, ' wheat')
                    done = true
                    reward = reward + 1.0 + 0.25 * info.wheat_collected
                end
            else
                return self:step(0) -- no-op for non-chest block
            end
        else
            print('Unknown action, ', action)
        end
    
        self.timesteps_remaining = self.timesteps_remaining - 1
        done = done or self.timesteps_remaining == 0
    
        return self.getObs(), reward, done, truncated, info
    end,

    getFacing = function(self)
        ret, data = turtle.inspect()
        if not ret then
            return nil
        end
    
        if data.name == 'minecraft:wheat' then
            return {
                name = 'wheat',
                age = data.state.age,
                solid = false
            }
        elseif data.name == 'minecraft:farmland' then
            return {
                name = 'farmland',
                solid = true
            }
        elseif data.name == 'minecraft:chest' then
            return {
                name = 'chest',
                solid = true
            }
        elseif data.tags then
            if data.tags.contains('forge:grass') then
                return {
                    name = 'grass',
                    solid = true
                }
            elseif data.tags.contains('forge:dirt') then
                return {
                    name = 'dirt',
                    solid = true
                }
            elseif data.tags.contains('forge:water') then
                return {
                    name = 'water',
                    solid = false
                }
            elseif data.tags.contains('forge:fence') then
                return {
                    name = 'fence',
                    solid = true
                }
            end
        end
    
        -- looks stupid to call air solid, but we'll use it as a placeholder
        -- for an unknown type, and we have to assume we cannot move through it
        return {
            name = 'air',
            solid = true
        }
    end,

    fuelCheck = function(self)
        -- automatically refuels turtle if needed
    
        if turtle.getFuelLevel() == 0 then
            print('Trying to refuel turtle')
            slot = turtle.getSelectedSlot()
            for i=1,16,1 do
                turtle.select(i)
                if turtle.refuel(1) then
                    break
                end
            end
            turtle.select(slot)
            print('New fuel level: ', turtle.getFuelLevel())
        end
    end,

    -- this returns as much of an observation as we can,
    -- but we'll have to do some post processing before we
    -- can pass it to rllib
    getObs = function(self)
        obs = {
            timesteps_remaining = self.timesteps_remaining / self.max_timesteps,
            world_time = ((os.time() + 18) % 24) * 1000,
            light_level = nil,
            fuel = turtle.getFuelLevel(),
            wheat = self:getWheat(),
            direction = self.tracker.dir,
            chest_displacement = self.tracker.pos - self.chestLoc,
            field_displacement = self.tracker.pos - self.fieldLoc,
            facing = nil,
            wheat_age = nil
        }

        block = self:getFacing()
        if block == nil then
            obs['facing'] = oneHot(self.blocks, block)
            obs['wheat_age'] = 0
        else
            obs['facing'] = oneHot(self.blocks, block.name)
            if block.age then
                obs['wheat_age'] = block.age / 7
            else
                obs['wheat_age'] = 0
            end
        end

        mask = {}
        for _, action in pairs(self.actions) do
            mask[action] = 1
        end

        if block and block.solid then
            mask['move-forward'] = 0
        end

        if not (block and block.age == 7) then
            mask['harvest'] = 0
        end

        if self.getWheat() == 0 or not (block and block.name == 'chest') then
            mask['interact-chest'] = 0
        end

        action_mask = {}
        for _, allowed in pairs(mask) do
            action_mask[#action_mask + 1] = allowed
        end

        return {
            action_mask = action_mask,
            observations = obs
        }
    end,

    getWheat = function(self)
        wheat = 0
        for i=1,16,1 do
            item = turtle.getItemDetail(i)
            if item and item.name == 'minecraft:wheat' then
                wheat = what + turtle.getItemCount(i)
            end
        end
    
        return wheat
    end
}

local ametatable = {
    __index = agent
}

function agent.new(timesteps, chestLoc, fieldLoc)
    return setmetatable({
        max_timesteps = timesteps,
        timesteps_remaining = timesteps,
        tracker = tracker,
        chestLoc = chestLoc,
        fieldLoc = fieldLoc
    }, ametatable)
end

return agent