#![allow(non_upper_case_globals)]
#![allow(dead_code)]

#[macro_use]
extern crate bitflags;
extern crate piston_window;
extern crate rand;
extern crate num_traits;
extern crate delaunator;

use piston_window::*;
mod map_gen;
use map_gen::*;

const WHITE: types::Color = [1.0, 1.0, 1.0, 1.0];
const BLACK: types::Color = [0.0, 0.0, 0.0, 1.0];
const PURPLE: types::Color = [1.0, 0.0, 1.0, 1.0];

const TILE_SIZE: f64 = 10.0;

fn main() {
    let mut window: PistonWindow = WindowSettings::new("Hello", [640, 480])
                                                  .vsync(true)
                                                  .exit_on_esc(true)
                                                  .build().unwrap();
    let map_width = 100;
    let map_height = 100;
    let draw_size = window.draw_size();
    // for centering
    let offset_left = draw_size.width / 2.0 - (map_width as f64 * TILE_SIZE) / 2.0;
    let offset_top = draw_size.height / 2.0 - (map_height as f64 * TILE_SIZE) / 2.0;
    // generate map
    let mut gen = MapGenerator::new(map_width, map_height);
    gen.fill(TileKind::Floor);
    gen.caverns();
    // this tile is reused for all tiles
    let mut rect: [f64; 4] = [0.0, 0.0, TILE_SIZE, TILE_SIZE];
    // event loop
    while let Some(event) = window.next() {
        match event {
            Event::Loop(loop_event) => {
                match loop_event {
                    Loop::Render(_args) => {
                        // render when render event is received
                        window.draw_2d(&event, |ctx, gfx, _dev| {
                            clear(WHITE, gfx);
                            // draw tiles
                            for x in 0..map_width {
                                for y in 0..map_height {
                                    let tile = gen.get_tile(x, y);
                                    let color: types::Color = match tile {
                                        TileKind::Floor => WHITE,
                                        TileKind::Wall => BLACK,
                                        _ => PURPLE
                                    };
                                    rect[0] = x as f64 * TILE_SIZE + offset_left;
                                    rect[1] = y as f64 * TILE_SIZE + offset_top;
                                    rectangle(color, rect, ctx.transform, gfx);
                                }
                            }
                        });
                    },
                    _ => {}
                }
            },
            Event::Input(input_event, _timestamp) => {
                match input_event {
                    Input::Button(ButtonArgs{state, button, scancode: _}) => {
                        match button {
                            Button::Mouse(mouse_button) => {
                                // regenerate map on click
                                if mouse_button == MouseButton::Left && state == ButtonState::Press {
                                    gen.fill(TileKind::Floor);
                                    gen.caverns();
                                }
                            },
                            _ => {}
                        }
                    },
                    _ => {}
                }
            },
            _ => {}
        }
    }
}