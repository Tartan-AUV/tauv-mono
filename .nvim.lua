vim.api.nvim_create_user_command("DeployOsprey", function()
	local cwd = vim.fn.getcwd()
	local remote = "tauv@10.0.0.20:/home/tauv/tauv-mono"
	local cmd = string.format("rsync -avz --exclude-from=%s/.syncignore %s/ %s", cwd, cwd, remote)
	vim.fn.jobstart(cmd, {
		stdout_buffered = true,
		on_stdout = function(_, data)
			if data then
				for _, line in ipairs(data) do
					print(line)
				end
			end
		end,
		on_stderr = function(_, data)
			if data then
				for _, line in ipairs(data) do
					vim.notify(line, vim.log.levels.ERROR)
				end
			end
		end,
		on_exit = function(_, code)
			if code == 0 then
				vim.notify("Rsync complete", vim.log.levels.INFO)
			else
				vim.notify("Rsync failed", vim.log.levels.ERROR)
			end
		end,
	})
end, {})
