use std::cmp;
use std::collections::{HashSet, VecDeque};
use std::ops::*;
use rand::prelude::*;
use num_traits::*;

fn abs<T: Signed + Zero + Ord>(n: T) -> T {
    if n < T::zero() {
        T::zero() - n
    } else {
        n
    }
}

fn manhattan_distance<T: Signed + Zero + Ord>(ax: T, ay: T, bx: T, by: T) -> T {
    return abs(ax - bx) + abs(ay - by);
}

fn linear_scale<T: Num + Copy>(n: T, src_min: T, src_max: T, dst_min: T, dst_max: T) -> T {
    return ((dst_max - dst_min) * (n - src_min) / (src_max - src_min)) + dst_min;
}

enum DirectionalRange {
    Forwards(std::ops::Range<isize>),
    Backwards(std::iter::Rev<std::ops::Range<isize>>)
}

impl Iterator for DirectionalRange {
    type Item = isize;

    fn next(&mut self) -> Option<isize> {
        match self {
            DirectionalRange::Forwards(range) => range.next(),
            DirectionalRange::Backwards(range) => range.next()
        }
    }
}

fn range(from: isize, to: isize) -> DirectionalRange {
    if to < from {
        DirectionalRange::Backwards((to .. from).rev())
    } else {
        DirectionalRange::Forwards(from .. to)
    }
}

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
struct Vec2 {
    x: isize,
    y: isize
}

impl From<&delaunator::Point> for Vec2 {
    fn from(point: &delaunator::Point) -> Self {
        return Vec2{
            x: point.x as isize,
            y: point.y as isize
        };
    }
}

impl From<&Vec2> for delaunator::Point {
    fn from(vec2: &Vec2) -> Self {
        return delaunator::Point {
            x: vec2.x as f64,
            y: vec2.y as f64
        };
    }
}

struct Node<T> {
    value: T,
    children: Vec<Node<T>>
}

impl<T> Node<T> {
    fn new(value: T) -> Node<T> {
        Node {
            value,
            children: Vec::new()
        }
    }

    fn add_child(&mut self, value: T) {
        self.children.push(Node::new(value));
    }
}

bitflags! {
    pub struct TileKind: u8 {
        const Floor = 0;
        const Wall = 1;
    }
}

pub struct MapGenerator {
    width: isize,
    height: isize,
    rng: ThreadRng,
    tiles: Vec<TileKind>,
    swap_tiles: Vec<TileKind>
}

impl MapGenerator {
    pub fn new(width: usize, height: usize) -> MapGenerator {
        MapGenerator {
            width: width as isize,
            height: height as isize,
            rng: rand::thread_rng(),
            tiles: vec![TileKind::Floor; width * height],
            swap_tiles: Vec::new()
        }
    }

    pub fn get_dimensions(&self) -> (isize, isize) {
        return (self.width, self.height);
    }

    fn get_index<T: NumCast>(&self, x: T, y: T) -> usize {
        let y = y.to_usize().unwrap();
        let x = x.to_usize().unwrap();
        let i = y * (self.width as usize) + x;
        return i;
    }

    pub fn get_tile<T: NumCast>(&self, x: T, y: T) -> TileKind {
        let i = self.get_index(x, y);
        return self.tiles[i];
    }

    fn set_tile<T: NumCast>(&mut self, x: T, y: T, value: TileKind) {
        let i = self.get_index(x, y);
        self.tiles[i] = value;
    }

    fn center_weighed_noise(&mut self, edge_freq: f64, center_freq: f64, value: TileKind) {
        let center_x = self.width as isize / 2;
        let center_y = self.height as isize / 2;
        let max_dist = manhattan_distance(0, 0, center_x, center_y);

        for x in 0 .. self.width as isize {
            for y in 0 .. self.height as isize {
                let dist = manhattan_distance(x, y, center_x, center_y);
                let freq = linear_scale(dist as f64, 0.0, max_dist as f64, center_freq, edge_freq);

                if self.rng.gen::<f64>() < freq {
                    self.set_tile(x, y, value);
                }
            }
        }
    }

    // fill the border with a value
    fn border(&mut self, thickness: isize, value: TileKind) {
        let thickness_h = cmp::min(self.width, thickness);
        let thickness_v = cmp::min(self.height, thickness);

        // top and bottom
        for x in 0 .. self.width {
            for t in 0 .. thickness_v {
                self.set_tile(x, t, value);
                self.set_tile(x, self.height - t - 1, value);
            }
        }

        // left and right
        // skip corners because they were already filled by the code above
        for y in thickness_v .. self.height - thickness_v {
            for t in 0 .. thickness_h {
                self.set_tile(t, y, value);
                self.set_tile(self.width - t - 1, y, value);
            }
        }
    }

    // count how many neighbors equal cmp_val
    fn count_neighbors_eq(&self, at_x: isize, at_y: isize, radius: isize, cmp_val: TileKind) -> isize {
        // clamp neighorhood dimensions to prevent underflow and wrapping
        let left = if radius > at_x { 0 } else { at_x - radius };
        let right = cmp::min(self.width - 1, at_x + radius);
        let top = if radius > at_y { 0 } else { at_y - radius };
        let bottom = cmp::min(self.height - 1, at_y + radius);

        let mut count = 0;

        // iterate neighbors
        for x in left ..= right {
            for y in top ..= bottom {
                if self.get_tile(x, y) == cmp_val {
                    count += 1;
                }
            }
        }

        // uncount self
        if self.get_tile(at_x, at_y) == cmp_val {
            count -= 1;
        }

        return count;
    }

    // gives the positions of neighbors to the receiver closure
    fn receive_neighbor_positions<F: FnMut(Vec2)>(&self, at_x: isize, at_y: isize, radius: isize, mut receiver: F) {
        // clamp neighorhood dimensions to prevent underflow and wrapping
        let left = if radius > at_x { 0 } else { at_x - radius };
        let right = cmp::min(self.width - 1, at_x + radius);
        let top = if radius > at_y { 0 } else { at_y - radius };
        let bottom = cmp::min(self.height - 1, at_y + radius);

        // iterate neighbors
        for x in left ..= right {
            for y in top ..= bottom {
                receiver(Vec2{x, y});
            }
        }
    }

    fn cellular_automaton<F: FnMut(&mut Self, isize, isize) -> TileKind>(&mut self, num_iter: isize, mut rule: F) {
        // init swap buffer
        if self.swap_tiles.len() != self.tiles.len() {
            self.swap_tiles.resize(self.tiles.len(), TileKind::Floor);
        }

        for _ in 0 .. num_iter {
            // run CA
            for x in 0 .. self.width {
                for y in 0 .. self.height {
                    let next_state = rule(self, x, y);
                    let i = self.get_index(x, y);
                    self.swap_tiles[i] = next_state;
                }
            }

            // swap buffers
            std::mem::swap(&mut self.tiles, &mut self.swap_tiles);
        }
    }

    fn line_helper_low(&mut self, from: Vec2, to: Vec2, value: TileKind) {
        let dx = to.x - from.x;
        let mut dy = to.y - from.y;
        let mut yi = 1;

        if dy < 0 {
            yi = -1;
            dy = -dy;
        }

        let mut d = dy * 2 - dx;
        let mut y = from.y;
        
        for x in range(from.x, to.x) {
            self.set_tile(x, y, value);

            if d > 0 {
                y += yi;
                d -= dx * 2;
            }

            d += dy * 2;
        }
    }

    fn line_helper_high(&mut self, from: Vec2, to: Vec2, value: TileKind) {
        let mut dx = to.x - from.x;
        let dy = to.y - from.y;
        let mut xi = 1;

        if dx < 0 {
            xi = -1;
            dx = -dx;
        }

        let mut d = dy * 2 - dx;
        let mut x = from.x;
        
        for y in range(from.y, to.y) {
            self.set_tile(x, y, value);

            if d > 0 {
                x += xi;
                d -= dy * 2;
            }

            d += dx * 2;
        }
    }

    fn line(&mut self, from: Vec2, to: Vec2, value: TileKind) {
        if abs(to.y - from.y) < abs(to.x - from.x) {
            if from.x > to.x {
                self.line_helper_low(to, from, value);
            } else {
                self.line_helper_low(from, to, value);
            }
        } else {
            if from.y > to.y {
                self.line_helper_high(from, to, value);
            } else {
                self.line_helper_high(to, from, value);
            }
        }
    }

    pub fn fill(&mut self, value: TileKind) {
        for i in 0 .. self.tiles.len() {
            self.tiles[i] = value;
        }
    }

    fn floodfill(&mut self, at_x: isize, at_y: isize, fill_kind: TileKind) -> Vec<Vec2> {
        let mut area_tiles: HashSet<Vec2> = HashSet::new();
        let mut frontier: VecDeque<Vec2> = VecDeque::new();

        let start = Vec2{x: at_x, y: at_y};
        frontier.push_front(start);

        while frontier.len() > 0 {
            let cur = frontier.pop_front().unwrap();
            let tile = self.get_tile(cur.x, cur.y);

            if tile == fill_kind && !area_tiles.contains(&cur) {
                area_tiles.insert(cur);
                self.receive_neighbor_positions(cur.x, cur.y, 1, |pos| {
                    frontier.push_front(pos);
                });
            }
        }

        return area_tiles.drain().collect();
    }

    fn find_areas(&mut self, find_kind: TileKind) -> Vec<Vec<Vec2>> {
        const unflagged: isize = 0;
        let mut area_counter: isize = 1;
        let mut tile_flags: Vec<isize> = Vec::new();
        tile_flags.resize(self.tiles.len(), unflagged);
        let mut areas: Vec<Vec<Vec2>> = Vec::new();

        // find areas
        for x in 0 .. self.width {
            for y in 0 .. self.height {
                let i = (y * self.width + x) as usize;
                // check if tile hasn't been flagged yet
                if self.tiles[i] == find_kind && tile_flags[i] == unflagged {
                    // flag area
                    let area_tiles = self.floodfill(x, y, find_kind);

                    // copy flags
                    for pos in &area_tiles {
                        tile_flags[self.get_index(pos.x, pos.y)] = area_counter;
                    }
                    // save area
                    areas.push(area_tiles);
                }
                area_counter += 1;
            }
        }

        return areas;
    }

    fn connect_areas<T>(&mut self, areas: &[T], connect_kind: TileKind)
        where T: AsRef<[Vec2]>
    {
        let connection_points: Vec<delaunator::Point> = areas.into_iter().map(|asref| {
            let tile_positions = asref.as_ref();
            let rand_tile = tile_positions[self.rng.gen_range(0, tile_positions.len())];
            return delaunator::Point::from(&rand_tile);
        }).collect();

        // minimum spanning tree of areas
        let mst: Option<Node<Vec2>> = match connection_points.len() {
            0 | 1 => {
                // nothing to connect
                None
            },
            2 => {
                // the graph of two points is the mst
                let mut root = Node::new(Vec2::from(&connection_points[0]));
                root.add_child(Vec2::from(&connection_points[1]));
                Some(root)
            },
            _ => {
                // create mst from triangulation
                let delaunay = delaunator::triangulate(&connection_points);
                None
            }
        };

        // connect areas
        if let Some(mst) = mst {
            let mut frontier: VecDeque<Node<Vec2>> = VecDeque::new();
            frontier.push_front(mst);

            while frontier.len() > 0 {
                let cur = frontier.pop_front().unwrap();

                for next in cur.children {
                    self.line(cur.value, next.value, connect_kind);
                    frontier.push_front(next);
                }
            }
        }
    }

    pub fn caverns(&mut self) {
        self.border(2, TileKind::Wall);

        // initialize CA with noise
        self.center_weighed_noise(0.8, 0.4, TileKind::Wall);

        // smooths the noise into cavern-like shapes
        self.cellular_automaton(3, |self2, x, y| {
            let cur_state = self2.tiles[self2.get_index(x, y)];
            let num_wall = self2.count_neighbors_eq(x, y, 1, TileKind::Wall);
            
            return if cur_state == TileKind::Wall {
                if num_wall > 3 {
                    TileKind::Wall
                } else {
                    TileKind::Floor
                }
            } else {
                if num_wall > 4 {
                    TileKind::Wall
                } else {
                    cur_state
                }
            };
        });

        // turn small patches of wall into floor
        for tile_positions in &self.find_areas(TileKind::Wall) {
            if tile_positions.len() < 10 {
                for pos in tile_positions {
                    self.set_tile(pos.x, pos.y, TileKind::Floor);
                }
            }
        }
        
        // turn too small caves into wall
        let caves: Vec<Vec<Vec2>> = self.find_areas(TileKind::Floor)
            .into_iter()
            .filter(|tile_positions| {
                if tile_positions.len() < 100 {
                    for pos in tile_positions {
                        self.set_tile(pos.x, pos.y, TileKind::Wall);
                    }
                    return false;
                }
                return true;
            })
            .collect();

        // connect caves
        self.connect_areas(&caves, TileKind::Floor);
    }
}
