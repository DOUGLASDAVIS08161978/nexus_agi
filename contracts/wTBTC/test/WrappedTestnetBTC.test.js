const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("WrappedTestnetBTC", function () {
  let wTBTC;
  let operator;
  let user1;
  let user2;
  let user3;

  beforeEach(async function () {
    [operator, user1, user2, user3] = await ethers.getSigners();

    const WrappedTestnetBTC = await ethers.getContractFactory("WrappedTestnetBTC");
    wTBTC = await WrappedTestnetBTC.deploy(operator.address);
    await wTBTC.waitForDeployment();
  });

  describe("Deployment", function () {
    it("Should set the correct token details", async function () {
      expect(await wTBTC.name()).to.equal("Wrapped Testnet Bitcoin");
      expect(await wTBTC.symbol()).to.equal("wTBTC");
      expect(await wTBTC.decimals()).to.equal(18);
    });

    it("Should set the deployer as bridge operator", async function () {
      expect(await wTBTC.bridgeOperator()).to.equal(operator.address);
    });

    it("Should start unpaused", async function () {
      expect(await wTBTC.paused()).to.equal(false);
    });

    it("Should start with zero total supply", async function () {
      expect(await wTBTC.totalSupply()).to.equal(0);
    });
  });

  describe("Minting", function () {
    it("Should mint tokens by bridge operator", async function () {
      const amount = ethers.parseEther("10");
      await expect(
        wTBTC.connect(operator).mint(user1.address, amount, "btc_tx_123")
      )
        .to.emit(wTBTC, "Mint")
        .withArgs(user1.address, amount, "btc_tx_123")
        .and.to.emit(wTBTC, "Transfer")
        .withArgs(ethers.ZeroAddress, user1.address, amount);

      expect(await wTBTC.balanceOf(user1.address)).to.equal(amount);
      expect(await wTBTC.totalSupply()).to.equal(amount);
    });

    it("Should revert if non-operator tries to mint", async function () {
      const amount = ethers.parseEther("10");
      await expect(
        wTBTC.connect(user1).mint(user2.address, amount, "btc_tx_123")
      ).to.be.revertedWith("Only bridge operator");
    });

    it("Should revert when minting to zero address", async function () {
      const amount = ethers.parseEther("10");
      await expect(
        wTBTC.connect(operator).mint(ethers.ZeroAddress, amount, "btc_tx_123")
      ).to.be.revertedWith("Mint to zero address");
    });

    it("Should revert when minting zero amount", async function () {
      await expect(
        wTBTC.connect(operator).mint(user1.address, 0, "btc_tx_123")
      ).to.be.revertedWith("Amount must be positive");
    });

    it("Should revert when paused", async function () {
      await wTBTC.connect(operator).setPaused(true);
      const amount = ethers.parseEther("10");
      await expect(
        wTBTC.connect(operator).mint(user1.address, amount, "btc_tx_123")
      ).to.be.revertedWith("Contract paused");
    });
  });

  describe("Burning", function () {
    beforeEach(async function () {
      // Mint some tokens first
      await wTBTC.connect(operator).mint(
        user1.address,
        ethers.parseEther("100"),
        "btc_tx_123"
      );
    });

    it("Should burn tokens and emit events", async function () {
      const amount = ethers.parseEther("50");
      const btcAddress = "tb1qtest123";

      await expect(wTBTC.connect(user1).burn(amount, btcAddress))
        .to.emit(wTBTC, "Burn")
        .withArgs(user1.address, amount, btcAddress)
        .and.to.emit(wTBTC, "Transfer")
        .withArgs(user1.address, ethers.ZeroAddress, amount);

      expect(await wTBTC.balanceOf(user1.address)).to.equal(ethers.parseEther("50"));
      expect(await wTBTC.totalSupply()).to.equal(ethers.parseEther("50"));
    });

    it("Should revert when burning more than balance", async function () {
      const amount = ethers.parseEther("200");
      await expect(
        wTBTC.connect(user1).burn(amount, "tb1qtest123")
      ).to.be.revertedWith("Insufficient balance");
    });

    it("Should revert when bitcoin address is empty", async function () {
      const amount = ethers.parseEther("50");
      await expect(
        wTBTC.connect(user1).burn(amount, "")
      ).to.be.revertedWith("Bitcoin address required");
    });

    it("Should revert when paused", async function () {
      await wTBTC.connect(operator).setPaused(true);
      const amount = ethers.parseEther("50");
      await expect(
        wTBTC.connect(user1).burn(amount, "tb1qtest123")
      ).to.be.revertedWith("Contract paused");
    });
  });

  describe("ERC20 Transfer", function () {
    beforeEach(async function () {
      await wTBTC.connect(operator).mint(
        user1.address,
        ethers.parseEther("100"),
        "btc_tx_123"
      );
    });

    it("Should transfer tokens between accounts", async function () {
      const amount = ethers.parseEther("50");

      await expect(wTBTC.connect(user1).transfer(user2.address, amount))
        .to.emit(wTBTC, "Transfer")
        .withArgs(user1.address, user2.address, amount);

      expect(await wTBTC.balanceOf(user1.address)).to.equal(ethers.parseEther("50"));
      expect(await wTBTC.balanceOf(user2.address)).to.equal(ethers.parseEther("50"));
    });

    it("Should revert when transferring more than balance", async function () {
      const amount = ethers.parseEther("200");
      await expect(
        wTBTC.connect(user1).transfer(user2.address, amount)
      ).to.be.revertedWith("Insufficient balance");
    });

    it("Should revert when transferring to zero address", async function () {
      const amount = ethers.parseEther("50");
      await expect(
        wTBTC.connect(user1).transfer(ethers.ZeroAddress, amount)
      ).to.be.revertedWith("Transfer to zero address");
    });

    it("Should revert when paused", async function () {
      await wTBTC.connect(operator).setPaused(true);
      const amount = ethers.parseEther("50");
      await expect(
        wTBTC.connect(user1).transfer(user2.address, amount)
      ).to.be.revertedWith("Contract paused");
    });
  });

  describe("ERC20 Approval", function () {
    it("Should approve tokens for spending", async function () {
      const amount = ethers.parseEther("50");

      await expect(wTBTC.connect(user1).approve(user2.address, amount))
        .to.emit(wTBTC, "Approval")
        .withArgs(user1.address, user2.address, amount);

      expect(await wTBTC.allowance(user1.address, user2.address)).to.equal(amount);
    });

    it("Should revert when paused", async function () {
      await wTBTC.connect(operator).setPaused(true);
      const amount = ethers.parseEther("50");
      await expect(
        wTBTC.connect(user1).approve(user2.address, amount)
      ).to.be.revertedWith("Contract paused");
    });
  });

  describe("ERC20 TransferFrom", function () {
    beforeEach(async function () {
      await wTBTC.connect(operator).mint(
        user1.address,
        ethers.parseEther("100"),
        "btc_tx_123"
      );
      await wTBTC.connect(user1).approve(user2.address, ethers.parseEther("50"));
    });

    it("Should transfer tokens using allowance", async function () {
      const amount = ethers.parseEther("30");

      await expect(
        wTBTC.connect(user2).transferFrom(user1.address, user3.address, amount)
      )
        .to.emit(wTBTC, "Transfer")
        .withArgs(user1.address, user3.address, amount);

      expect(await wTBTC.balanceOf(user1.address)).to.equal(ethers.parseEther("70"));
      expect(await wTBTC.balanceOf(user3.address)).to.equal(ethers.parseEther("30"));
      expect(await wTBTC.allowance(user1.address, user2.address)).to.equal(
        ethers.parseEther("20")
      );
    });

    it("Should revert when exceeding allowance", async function () {
      const amount = ethers.parseEther("60");
      await expect(
        wTBTC.connect(user2).transferFrom(user1.address, user3.address, amount)
      ).to.be.revertedWith("Allowance exceeded");
    });

    it("Should revert when paused", async function () {
      await wTBTC.connect(operator).setPaused(true);
      const amount = ethers.parseEther("30");
      await expect(
        wTBTC.connect(user2).transferFrom(user1.address, user3.address, amount)
      ).to.be.revertedWith("Contract paused");
    });
  });

  describe("Bridge Operator Management", function () {
    it("Should change bridge operator", async function () {
      await expect(wTBTC.connect(operator).changeBridgeOperator(user1.address))
        .to.emit(wTBTC, "BridgeOperatorChanged")
        .withArgs(operator.address, user1.address);

      expect(await wTBTC.bridgeOperator()).to.equal(user1.address);
    });

    it("Should revert when changing to zero address", async function () {
      await expect(
        wTBTC.connect(operator).changeBridgeOperator(ethers.ZeroAddress)
      ).to.be.revertedWith("Zero address");
    });

    it("Should revert when non-operator tries to change operator", async function () {
      await expect(
        wTBTC.connect(user1).changeBridgeOperator(user2.address)
      ).to.be.revertedWith("Only bridge operator");
    });

    it("Should allow new operator to mint after change", async function () {
      await wTBTC.connect(operator).changeBridgeOperator(user1.address);

      const amount = ethers.parseEther("10");
      await expect(
        wTBTC.connect(user1).mint(user2.address, amount, "btc_tx_456")
      ).to.not.be.reverted;

      expect(await wTBTC.balanceOf(user2.address)).to.equal(amount);
    });

    it("Should prevent old operator from minting after change", async function () {
      await wTBTC.connect(operator).changeBridgeOperator(user1.address);

      const amount = ethers.parseEther("10");
      await expect(
        wTBTC.connect(operator).mint(user2.address, amount, "btc_tx_456")
      ).to.be.revertedWith("Only bridge operator");
    });
  });

  describe("Pause Functionality", function () {
    it("Should pause contract", async function () {
      await expect(wTBTC.connect(operator).setPaused(true))
        .to.emit(wTBTC, "Paused")
        .withArgs(true);

      expect(await wTBTC.paused()).to.equal(true);
    });

    it("Should unpause contract", async function () {
      await wTBTC.connect(operator).setPaused(true);

      await expect(wTBTC.connect(operator).setPaused(false))
        .to.emit(wTBTC, "Paused")
        .withArgs(false);

      expect(await wTBTC.paused()).to.equal(false);
    });

    it("Should revert when non-operator tries to pause", async function () {
      await expect(
        wTBTC.connect(user1).setPaused(true)
      ).to.be.revertedWith("Only bridge operator");
    });

    it("Should block all operations when paused", async function () {
      await wTBTC.connect(operator).mint(
        user1.address,
        ethers.parseEther("100"),
        "btc_tx_123"
      );
      await wTBTC.connect(operator).setPaused(true);

      // Test all operations are blocked
      await expect(
        wTBTC.connect(operator).mint(user1.address, ethers.parseEther("10"), "tx")
      ).to.be.revertedWith("Contract paused");

      await expect(
        wTBTC.connect(user1).burn(ethers.parseEther("10"), "tb1qtest")
      ).to.be.revertedWith("Contract paused");

      await expect(
        wTBTC.connect(user1).transfer(user2.address, ethers.parseEther("10"))
      ).to.be.revertedWith("Contract paused");

      await expect(
        wTBTC.connect(user1).approve(user2.address, ethers.parseEther("10"))
      ).to.be.revertedWith("Contract paused");
    });
  });

  describe("Complex Scenarios", function () {
    it("Should handle multiple mints correctly", async function () {
      await wTBTC.connect(operator).mint(user1.address, ethers.parseEther("10"), "tx1");
      await wTBTC.connect(operator).mint(user1.address, ethers.parseEther("20"), "tx2");
      await wTBTC.connect(operator).mint(user2.address, ethers.parseEther("30"), "tx3");

      expect(await wTBTC.balanceOf(user1.address)).to.equal(ethers.parseEther("30"));
      expect(await wTBTC.balanceOf(user2.address)).to.equal(ethers.parseEther("30"));
      expect(await wTBTC.totalSupply()).to.equal(ethers.parseEther("60"));
    });

    it("Should handle full cycle: mint -> transfer -> burn", async function () {
      // Mint
      await wTBTC.connect(operator).mint(
        user1.address,
        ethers.parseEther("100"),
        "btc_tx_123"
      );

      // Transfer
      await wTBTC.connect(user1).transfer(user2.address, ethers.parseEther("40"));

      // Burn from user1
      await wTBTC.connect(user1).burn(ethers.parseEther("30"), "tb1qtest1");

      // Burn from user2
      await wTBTC.connect(user2).burn(ethers.parseEther("20"), "tb1qtest2");

      expect(await wTBTC.balanceOf(user1.address)).to.equal(ethers.parseEther("30"));
      expect(await wTBTC.balanceOf(user2.address)).to.equal(ethers.parseEther("20"));
      expect(await wTBTC.totalSupply()).to.equal(ethers.parseEther("50"));
    });

    it("Should handle complex allowance and transferFrom scenario", async function () {
      // Mint to user1
      await wTBTC.connect(operator).mint(
        user1.address,
        ethers.parseEther("100"),
        "btc_tx_123"
      );

      // User1 approves user2 for 50 tokens
      await wTBTC.connect(user1).approve(user2.address, ethers.parseEther("50"));

      // User2 transfers 30 tokens from user1 to user3
      await wTBTC.connect(user2).transferFrom(
        user1.address,
        user3.address,
        ethers.parseEther("30")
      );

      // User2 transfers remaining 20 tokens from user1 to themselves
      await wTBTC.connect(user2).transferFrom(
        user1.address,
        user2.address,
        ethers.parseEther("20")
      );

      expect(await wTBTC.balanceOf(user1.address)).to.equal(ethers.parseEther("50"));
      expect(await wTBTC.balanceOf(user2.address)).to.equal(ethers.parseEther("20"));
      expect(await wTBTC.balanceOf(user3.address)).to.equal(ethers.parseEther("30"));
      expect(await wTBTC.allowance(user1.address, user2.address)).to.equal(0);
    });
  });
});
