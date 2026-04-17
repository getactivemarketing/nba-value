import { Config } from '@remotion/cli/config';

Config.setVideoImageFormat('jpeg');
Config.setConcurrency(1);
Config.setPixelFormat('yuv420p');
Config.setCodec('h264');
Config.setCrf(18);
