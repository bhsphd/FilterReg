//
// Created by wei on 9/10/18.
//

#pragma once

//From drake
#define POSER_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Classname)       \
  Classname(const Classname&) = default;                        \
  Classname& operator=(const Classname&) = default;             \
  Classname(Classname&&) = default;                    \
  Classname& operator=(Classname&&) = default;         \
  /* Fails at compile-time if default-copy doesn't work. */     \
  static void DRAKE_COPYABLE_DEMAND_COPY_CAN_COMPILE() {        \
    (void) static_cast<Classname& (Classname::*)(               \
        const Classname&)>(&Classname::operator=);              \
  }


#define POSER_DEFAULT_CONSTRUCT_DESTRUCT(ClassName)  \
  ClassName() = default;                             \
  ~ClassName() = default

//This one doesn't allow move
#define POSER_NO_COPY_ASSIGN_MOVE(ClassName)     \
  ClassName(const ClassName&) = delete;          \
  ClassName(ClassName&&) = delete;               \
  void operator=(const ClassName&) = delete;     \
  void operator=(ClassName&&) = delete

//This one allow default move
#define POSER_NO_COPY_ASSIGN(ClassName)      \
  ClassName(const ClassName&) = delete;      \
  void operator=(const ClassName&) = delete;
  