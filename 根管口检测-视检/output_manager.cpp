#include "output_manager.h"
#include <iostream>

OutputManager::OutputManager(const std::string& rootDir)
	: rootDir_(rootDir)
{
	if (!fs::exists(rootDir_))
		fs::create_directories(rootDir_);
}

std::string OutputManager::createImageDir(const std::string& imagePath)
{
	fs::path p(imagePath);
	std::string stem = p.stem().string();
	std::string dirPath = rootDir_ + "/" + stem;
	if (!fs::exists(dirPath))
		fs::create_directories(dirPath);
	return dirPath;
}

std::string OutputManager::getPath(const std::string& dir,
								   const std::string& filename)
{
	return dir + "/" + filename;
}
