library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package migen_helpers is
function to_std_ulogic(v: boolean) return std_ulogic;
function to_std_ulogic(v: integer) return std_ulogic;
function to_unsigned(v: std_ulogic; length: natural) return unsigned;
function get_index(v: unsigned; index: natural) return std_logic;

procedure clk_gen(
    signal clk : out std_logic;
    constant half_period : time;
--    signal run : in std_logic;
    constant first_edge_advance : time := 0 fs;
    constant initial_level: std_logic := '0'
);
end migen_helpers;

package body migen_helpers is
function to_std_ulogic(v: boolean) return std_ulogic is
begin
  if v then
    return '1';
  else
    return '0';
  end if;
end to_std_ulogic;

function to_std_ulogic(v: integer) return std_ulogic is
begin
  if v /= 0 then
    return '1';
  else
    return '0';
  end if;
end to_std_ulogic;

function to_unsigned(v: std_ulogic; length:natural) return unsigned is
begin
  if (v = '1') or (v = 'H') then
    return resize(unsigned'("1"),length);
  else
    return resize(unsigned'("0"),length);
  end if;
end to_unsigned;

function get_index(v: unsigned; index: natural) return std_logic is
begin
  return v(index);
end get_index;

procedure clk_gen(
    signal clk : out std_logic;
    constant half_period : time;
--    signal run : in std_logic;
    constant first_edge_advance : time := 0 fs;
    constant initial_level: std_logic := '0'
) is
begin
  -- Check the arguments
  assert (half_period /= 0 fs) report "clk_gen: half_period is zero; time resolution to large for frequency" severity FAILURE;
  -- Initial phase shift
  clk <= initial_level;
  wait for half_period - first_edge_advance;
  if initial_level /= '0' then
    clk <= '0';
    wait for half_period;
  end if;
  -- Generate cycles
  loop
    -- Only high pulse if run is '1' or 'H'
--    if (run = '1') or (run = 'H') then
--      clk <= run;
--    end if;
    clk <= '1';
    wait for half_period;
    -- Low part of cycle
    clk <= '0';
    wait for half_period;
  end loop;
end procedure;

end migen_helpers;