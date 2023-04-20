-- Modified from https://computercraft.info/wiki/Turtle_GPS_self-tracker_expansion_(tutorial)

local tracker

local function setLocation(self) -- get gps using other computers
    self.pos = vector.new(gps.locate())

    -- now move a small displacement to get our direction
    turtle.forward()
    newPos = vector.new(gps.locate())
    self.dir = newPos - self.pos
    turtle.back()

    self.cal = true
end

local function manSetLocation(self, x, y, z) -- manually set location
    self.pos = vector.new(x, y, z)
    self.cal = true
end

local function getLocation(self) -- return the location
    if self.cal then
        return self.pos
    else
        return nil
    end
end

local function turnLeft(self) -- turn left
    turtle.turnLeft()
    self.dir.x, self.dir.z = self.dir.z, -self.dir.x
end

local function turnRight(self) -- turn right
    turtle.turnRight()
    self.dir.x, self.dir.z = -self.dir.z, self.dir.x
end

local function forward(self) -- go forward
    turtle.forward()
    if self.cal then
        self.pos = self.pos + self.dir
    else
        print("Not Calibrated.")
    end
end

local function up(self) -- go up
    turtle.up()
    if self.cal then
        self.pos = self.pos + vector.new(0, 1, 0)
    else
        print("Not Calibrated.")
    end
end

local function down(self) -- go down
    turtle.down()
    if self.cal then
        self.pos = self.pos - vector.new(0, 1, 0)
    else
        print("Not Calibrated.")
    end
end

tracker = {
    pos = nil,
    cal = false,
    dir = vector.new(1, 0, 0),

    setLocation = setLocation,
    manSetLocation = manSetLocation,
    getLocation = getLocation,
    turnLeft = turnLeft,
    turnRight = turnRight,
    forward = forward,
    up = up,
    down = down
}

return tracker