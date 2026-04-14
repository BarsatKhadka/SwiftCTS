
set clk_period $::env(CLOCK_PERIOD)
set clk_port   $::env(CLOCK_PORT)

# 1. Clock 
create_clock -name $clk_port -period $clk_period [get_ports $clk_port]

# uncertainty due to jitter. give margin of error of 0.5
set_clock_uncertainty 0.5 [get_clocks $clk_port]

# first slew rise from 0.15 , max slew allowed 1.5 
set_clock_transition 0.15 [get_clocks $clk_port]
set_max_transition 1.5 [current_design]

#seperate clock from input ports 
set input_ports [get_ports -filter {direction == in && name != $clk_port}]
set output_ports [all_outputs]

# input delays
set_input_delay -min 0.5 -clock [get_clocks $clk_port] $input_ports
set_input_delay -max 2.0 -clock [get_clocks $clk_port] $input_ports    


#output delays , max means you are giving that "extra 2ns" to the Outside World, not to your chip.
set_output_delay -max 2.0 -clock [get_clocks $clk_port] [all_outputs]
set_output_delay -min 0.5 -clock [get_clocks $clk_port] [all_outputs]

# 5. driving cell as buffer_2 (medium) to simulate power source for inputs and outside pins see a capacitance load of 30pF
set_driving_cell -lib_cell sky130_fd_sc_hd__buf_2 $input_ports
set_load 0.03 $output_ports