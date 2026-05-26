function results = runEDPISLAM(dataFile, varargin)
% Usage:
%   runEDPISLAM
%   runEDPISLAM('..\TestData.mat')
%   runEDPISLAM('..\TestData.mat', 'NumAgents', 3, 'EnablePlot', true)

if nargin < 1 || isempty(dataFile)
    dataFile = fullfile(fileparts(mfilename('fullpath')), 'TestData.mat');
end

parser = inputParser;
parser.addParameter('NumAgents', inf, @(x) isnumeric(x) && isscalar(x) && x >= 1);
parser.addParameter('EnablePlot', false, @(x) islogical(x) || isnumeric(x));
parser.addParameter('ShowProgress', false, @(x) islogical(x) || isnumeric(x));
parser.addParameter('SaveDir', fullfile(fileparts(mfilename('fullpath')), 'results'), @ischar);
parser.parse(varargin{:});
opt = parser.Results;

if ~exist(dataFile, 'file')
    error('Data file not found: %s', dataFile);
end

if ~exist(opt.SaveDir, 'dir')
    mkdir(opt.SaveDir);
end

load(dataFile);
numAgents = numel(agents);
stepCounts = arrayfun(@(x) numel(x.stepLength), agents);
systemStep = max(stepCounts);


position_uncertainty_std = [0.1; 0.1; 1.0];
stephead_noise_std = [0.1; 1.0];
MapUpdateLate = 30;
Para_gridmatch = 1.5;
Para_optiThre = 30;
ParticleSaveNum = 10;
SampleNum = 2;

rawTraj = cell(1, numAgents);
for a = 1:numAgents
    rawTraj{a} = agents(a).initialPose;
    for k = 1:numel(agents(a).headChange)
        rawTraj{a}(:, k + 1) = PedestrianDeadReckoning(rawTraj{a}(:, k), ...
            agents(a).headChange(k), agents(a).stepLength(k));
    end
end

if logical(opt.EnablePlot)
    figure(10);
    clf;
    hold on;
    axis equal;
    for a = 1:numAgents
        plot(rawTraj{a}(1, :), rawTraj{a}(2, :), 'LineWidth', 2);
    end
    legend({agents.id}, 'Location', 'best');
end

particles = cell(1, numAgents);
pedInertialGrid = cell(1, numAgents);
pedGridIndex = ones(1, numAgents);
MarginGrid = cell(1, numAgents);
MarginCovInv = cell(1, numAgents);
MarginCovDet = cell(1, numAgents);
OptiBool = zeros(1, numAgents);

for a = 1:numAgents
    p0 = initParticle(agents(a).initialPose, position_uncertainty_std, numAgents);
    particles{a} = p0;
    pedInertialGrid{a} = [];

    Xestimated = agents(a).initialPose;
    Pestimated = diag(position_uncertainty_std .^ 2);
    MarginGrid{a}(:, 1) = Xestimated(1:2);
    MarginCovInv{a}(:, :, 1) = inv(Pestimated(1:2, 1:2) + eye(2));
    MarginCovDet{a}(1) = det(Pestimated(1:2, 1:2) + eye(2)) .^ 0.5;
end

tic;
for t = 1:systemStep
    if logical(opt.ShowProgress) && (mod(t, 200) == 0 || t == systemStep)
        fprintf('Progress: %d/%d\n', t, systemStep);
    end

    for a = 1:numAgents
        if t > stepCounts(a)
            continue;
        end

        StepHeadChange = [agents(a).stepLength(t), agents(a).headChange(t)];
        for i = 1:numel(particles{a})
            part = particles{a}(i);
            [part.Xestimated, part.Pestimated] = EKF_TimeUpdate( ...
                part.Xestimated, part.Pestimated, StepHeadChange, stephead_noise_std);
            part.Xall(:, t + 1) = part.Xestimated;

            
            if isempty(pedInertialGrid{a})
                MarginGrid{a}(:, t + 1) = part.Xestimated(1:2);
                MarginCovInv{a}(:, :, t + 1) = inv(part.Pestimated(1:2, 1:2) + eye(2));
                MarginCovDet{a}(t + 1) = det(part.Pestimated(1:2, 1:2) + eye(2)) .^ 0.5;
            end

            if t > MapUpdateLate
                InertialGrid = part.Xall(1:2, 1:t - MapUpdateLate);
                DistanceGrid2Posi = (InertialGrid - part.Xestimated(1:2)) .* ...
                    (inv(part.Pestimated(1:2, 1:2)) * (InertialGrid - part.Xestimated(1:2)));
                DistanceGrid2Posi = (sum(DistanceGrid2Posi)) .^ 0.5;

                GridMatch = find(DistanceGrid2Posi < 3);
                if ~isempty(GridMatch)
                    GridMatch(2, :) = t + 1;
                    part.GridMatch = [part.GridMatch, GridMatch];
                    OptiBool(a) = OptiBool(a) + 1;
                end
            end

            for b = 1:numAgents
                if b == a
                    continue;
                end
                if isempty(pedInertialGrid{b})
                    InertialGrid = MarginGrid{b};
                    useMargin = true;
                else
                    InertialGrid = pedInertialGrid{b};
                    useMargin = false;
                end

                DistanceGrid2Posi = (InertialGrid - part.Xestimated(1:2)) .* ...
                    (inv(part.Pestimated(1:2, 1:2)) * (InertialGrid - part.Xestimated(1:2)));
                DistanceGrid2Posi = (sum(DistanceGrid2Posi)) .^ 0.5;

                GridMatch = find(DistanceGrid2Posi < 3);
                if ~isempty(GridMatch)
                    GridMatch(2, :) = t + 1;
                    if useMargin
                        part.MarginMatch{b} = [part.MarginMatch{b}, GridMatch];
                    else
                        part.Match{b} = [part.Match{b}, GridMatch];
                    end
                end
            end

            particles{a}(i) = part;
        end
    end

    for a = 1:numAgents
        [particles{a}, pedInertialGrid{a}, pedGridIndex(a), OptiBool(a)] = optimizeAgentBlock( ...
            a, t, particles{a}, OptiBool(a), agents(a).stepLength, agents(a).headChange, ...
            position_uncertainty_std, stephead_noise_std, Para_gridmatch, SampleNum, ...
            ParticleSaveNum, Para_optiThre, pedInertialGrid, ...
            MarginGrid, MarginCovInv, MarginCovDet, numAgents, pedGridIndex(a));
    end
end
runtimeSeconds = toc;

correctedTraj = cell(1, numAgents);
for a = 1:numAgents
    correctedTraj{a} = particles{a}(1).Xall;
end

errorTable = struct('id', {}, 'rawError', {}, 'correctedError', {});
for a = 1:numAgents
    [rawErr, correctedErr] = computeError(rawTraj{a}, correctedTraj{a}, agents(a).matchResult);
    errorTable(a).id = agents(a).id;
    errorTable(a).rawError = rawErr;
    errorTable(a).correctedError = correctedErr;
    if isnan(correctedErr)
        fprintf('Mean localization error %s: N/A (missing matchResult)\n', agents(a).id);
    else
        fprintf('Mean localization error %s: %.4f (raw: %.4f)\n', ...
            agents(a).id, correctedErr, rawErr);
    end
end

fprintf('Computation finished. Steps: %d, runtime: %.4f s\n', systemStep, runtimeSeconds);

timeTag = datestr(now, 'yyyymmdd_HHMMSS');
resultFile = fullfile(opt.SaveDir, ['ColIOSLAM_results_', timeTag, '.mat']);
save(resultFile, 'agents', 'rawTraj', 'correctedTraj', 'errorTable', ...
    'runtimeSeconds', 'systemStep', 'particles', 'dataFile');
fprintf('Results saved to: %s\n', resultFile);

results = struct();
results.agents = agents;
results.rawTraj = rawTraj;
results.correctedTraj = correctedTraj;
results.errorTable = errorTable;
results.runtimeSeconds = runtimeSeconds;
results.resultFile = resultFile;
results.dataFile = dataFile;

end

function p = initParticle(initialPose, position_uncertainty_std, numAgents)
p.Xestimated = initialPose;
p.Pestimated = diag(position_uncertainty_std .^ 2);
p.Xall = initialPose;
p.GridMatch = [];
p.MarginMatch = cell(1, numAgents);
p.Match = cell(1, numAgents);
p.OptIniPose = initialPose;
p.OptIniInd = 1;
p.Weight = 0;
p.IndexOptiBegin = 1;
end

function [particlesA, pedGridA, pedIndexA, optiBoolA] = optimizeAgentBlock( ...
    a, t, particlesA, optiBoolA, stepLengthA, headChangeA, ...
    position_uncertainty_std, stephead_noise_std, Para_gridmatch, SampleNum, ...
    ParticleSaveNum, Para_optiThre, pedInertialGrid, ...
    MarginGrid, MarginCovInv, MarginCovDet, numAgents, pedIndexA)

if optiBoolA / numel(particlesA) <= Para_optiThre
    pedGridA = pedInertialGrid{a};
    return;
end
optiBoolA = 0;

ParticleIndex = 1;
NewParticle = struct([]);
SaveOptiValue = [];

for i = 1:numel(particlesA)
    baseParticle = particlesA(i);
    EndPoseMid = EndPoseEstimate(baseParticle.Xestimated(1:2), ...
        baseParticle.Pestimated(1:2, 1:2), Para_gridmatch / SampleNum);

    for EndPoseIndex = 1:size(EndPoseMid, 2)
        newParticle = baseParticle;
        [EndPose, EndPoseSigma] = KF_Measure(baseParticle.Xestimated, ...
            baseParticle.Pestimated, EndPoseMid(:, EndPoseIndex), ...
            diag(position_uncertainty_std(1:2) .^ 2));

        newParticle.Pestimated = EndPoseSigma;
        newParticle.OptIniPose = [baseParticle.OptIniPose(:, end), EndPose];
        newParticle.OptIniInd = [baseParticle.OptIniInd(end), t + 1];

        TrajAllMid = LocalInvKF(baseParticle.Xall, newParticle.OptIniPose, ...
            newParticle.OptIniInd, stepLengthA, headChangeA, stephead_noise_std);
        newParticle.Xestimated = TrajAllMid(:, end);
        newParticle.Xall = TrajAllMid;

        OptiValue = computeTotalOptiValue(TrajAllMid, newParticle, a, numAgents, ...
            pedInertialGrid, MarginGrid, MarginCovInv, MarginCovDet);
        newParticle.Weight = newParticle.Weight + OptiValue;

        if ParticleIndex == 1
            NewParticle = newParticle;
        else
            NewParticle(ParticleIndex) = newParticle;
        end
        SaveOptiValue(ParticleIndex) = newParticle.Weight;
        ParticleIndex = ParticleIndex + 1;
    end
end

[SaveOptiValue, SortIndex] = sort(SaveOptiValue);
NewParticle = NewParticle(SortIndex);
keepNum = min(numel(NewParticle), ParticleSaveNum);
particlesA = NewParticle(1:keepNum);
SaveOptiValue = SaveOptiValue(1:keepNum);

for i = 1:numel(particlesA)
    part = particlesA(i);
    TrajUpd = TrajOpti(part.Xall, Para_gridmatch, part.IndexOptiBegin, ...
        part.GridMatch, stepLengthA, headChangeA);
    part.IndexOptiBegin = t + 1;

    OptiValue = computeTotalOptiValue(TrajUpd, part, a, numAgents, ...
        pedInertialGrid, MarginGrid, MarginCovInv, MarginCovDet);
    if OptiValue < SaveOptiValue(i)
        part.Xall = TrajUpd;
    end

    part.GridMatch = [];
    for b = 1:numAgents
        if b ~= a
            part.MarginMatch{b} = [];
            part.Match{b} = [];
        end
    end
    particlesA(i) = part;
end

PoseX = [];
PoseY = [];
for i = 1:numel(particlesA)
    PoseX = [PoseX; particlesA(i).Xall(1, :)];
    PoseY = [PoseY; particlesA(i).Xall(2, :)];
end
pedGridA = [mean(PoseX(:, 1:pedIndexA - 1)); mean(PoseY(:, 1:pedIndexA - 1))];
for i = pedIndexA:size(PoseX, 2)
    if det(cov(PoseX(:, i), PoseY(:, i))) < 0.01
        pedGridA = [pedGridA, [mean(PoseX(:, pedIndexA)); mean(PoseY(:, pedIndexA))]];
        pedIndexA = pedIndexA + 1;
    else
        break;
    end
end
end

function OptiValue = computeTotalOptiValue(Traj, part, a, numAgents, ...
    pedInertialGrid, MarginGrid, MarginCovInv, MarginCovDet)
OptiValue = OptiFunValue(Traj, part.GridMatch);
for b = 1:numAgents
    if b == a
        continue;
    end
    if isempty(pedInertialGrid{b})
        OptiValueOther = ColFunValue(Traj, MarginGrid{b}, MarginCovInv{b}, ...
            MarginCovDet{b}, part.MarginMatch{b});
    else
        OptiValueOther = ColFunValue(Traj, pedInertialGrid{b}, [], [], part.Match{b});
    end
    OptiValue = OptiValue + OptiValueOther;
end
end

function [rawErr, correctedErr] = computeError(rawTraj, correctedTraj, matchResult)
rawErr = nan;
correctedErr = nan;
if isempty(matchResult)
    return;
end
idx = round(matchResult(1, :));
valid = idx >= 1 & idx <= size(correctedTraj, 2);
if ~any(valid)
    return;
end
idx = idx(valid);
truth = matchResult(2:3, valid);

rawDelta = rawTraj(1:2, idx) - truth;
correctedDelta = correctedTraj(1:2, idx) - truth;
rawErr = sum((sum(rawDelta .^ 2, 1)) .^ 0.5) / size(truth, 2);
correctedErr = sum((sum(correctedDelta .^ 2, 1)) .^ 0.5) / size(truth, 2);
end

function agents = loadAgentsData(dataFile, numAgentsLimit)
S = load(dataFile);

if isfield(S, 'agents')
    agents = S.agents;
else
    agents = convertLegacyData(S);
end

if isfinite(numAgentsLimit)
    agents = agents(1:min(numel(agents), floor(numAgentsLimit)));
end

for i = 1:numel(agents)
    agents(i) = normalizeAgent(agents(i), i);
end
end

function agents = convertLegacyData(S)
fields = fieldnames(S);
ids = [];
for i = 1:numel(fields)
    f = fields{i};
    if startsWith(f, 'initialPoseNo')
        idStr = extractAfter(string(f), 'initialPoseNo');
        idNum = str2double(idStr);
        if ~isnan(idNum)
            ids(end + 1) = idNum; %#ok<AGROW>
        end
    end
end
ids = sort(unique(ids));

if isempty(ids)
    error('No agents struct or legacy initialPoseNoX fields found.');
end

agents = struct('id', {}, 'initialPose', {}, 'stepLength', {}, 'headChange', {}, 'matchResult', {});
for i = 1:numel(ids)
    n = ids(i);
    fp = sprintf('initialPoseNo%d', n);
    fs = sprintf('stepLengthNo%d', n);
    fh = sprintf('headChangeNo%d', n);
    fm = sprintf('MatchResultNo%d', n);
    if ~(isfield(S, fp) && isfield(S, fs) && isfield(S, fh))
        continue;
    end
    a.id = sprintf('No%d', n);
    a.initialPose = S.(fp);
    a.stepLength = S.(fs);
    a.headChange = S.(fh);
    if isfield(S, fm)
        a.matchResult = S.(fm);
    else
        a.matchResult = [];
    end
    agents(end + 1) = a; %#ok<AGROW>
end

if isempty(agents)
    error('Legacy data exists but required fields are incomplete.');
end
end

function a = normalizeAgent(a, idx)
if ~isfield(a, 'id') || isempty(a.id)
    a.id = sprintf('No%d', idx);
else
    a.id = char(string(a.id));
end

required = {'initialPose', 'stepLength', 'headChange'};
for i = 1:numel(required)
    if ~isfield(a, required{i})
        error('Agent %s missing field: %s', a.id, required{i});
    end
end

if ~isfield(a, 'matchResult')
    a.matchResult = [];
end

a.initialPose = reshape(a.initialPose, 3, 1);
a.stepLength = reshape(a.stepLength, 1, []);
a.headChange = reshape(a.headChange, 1, []);
if numel(a.stepLength) ~= numel(a.headChange)
    error('Agent %s has inconsistent stepLength/headChange length.', a.id);
end
end
